"""
Run Optional Evaluations: Absorption, SCR, TPP, Unlearning

Note: Output directories must be provided even if evaluation is disabled.
Disabled evaluations will be skipped based on params.yaml settings.
"""
import os
import sys
import yaml
import pickle
import fire
from pathlib import Path

import sae_bench.custom_saes.run_all_evals_custom_saes as run_all_evals_custom_saes


def load_params():
    """Load parameters from params.yaml"""
    params_path = Path(__file__).parent.parent / "params.yaml"
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)


def load_saes(saes_dir: str):
    """Load SAE objects and create selected_saes list"""
    print("\n=== Loading SAEs ===")

    sae_dir = Path(saes_dir)

    # Load custom SAEs
    custom_sae_list_path = sae_dir / "custom_sae_list.txt"
    with open(custom_sae_list_path, 'r') as f:
        custom_sae_ids = [line.strip() for line in f]

    custom_saes = []
    for sae_id in custom_sae_ids:
        sae_path = sae_dir / f"{sae_id}.pkl"
        with open(sae_path, 'rb') as f:
            sae = pickle.load(f)
        custom_saes.append((sae_id, sae))
        print(f"Loaded custom SAE: {sae_id}")

    # Load comparison SAEs if available
    comparison_path = sae_dir / "comparison_saes.pkl"
    baseline_saes = []
    if comparison_path.exists():
        with open(comparison_path, 'rb') as f:
            baseline_saes = pickle.load(f)
        print(f"Loaded {len(baseline_saes)} comparison SAE(s)")

    selected_saes = custom_saes + baseline_saes
    print(f"\nTotal SAEs to evaluate: {len(selected_saes)}")

    return selected_saes


def get_enabled_eval_types(params):
    """Get list of enabled optional evaluation types"""
    optional_types = ['absorption', 'scr', 'tpp', 'unlearning', 'autointerp']
    enabled = [
        eval_type for eval_type in optional_types
        if params['eval_types'].get(eval_type, False)
    ]
    return enabled


def run_optional_evaluations(selected_saes, params, device, eval_types, output_dirs):
    """Run optional evaluations on selected SAEs

    Args:
        selected_saes: List of SAE tuples
        params: Loaded parameters
        device: CUDA device
        eval_types: List of enabled eval types
        output_dirs: Dict mapping eval_type -> output_dir
    """
    print("\n=== Running Optional Evaluations ===")
    print(f"Enabled evaluations: {eval_types}")

    if 'autointerp' in eval_types:
        print("\nWARNING: autointerp must be run as a script, not in notebook")
        eval_types.remove('autointerp')

    if not eval_types:
        print("No optional evaluations enabled")
        return None

    model_name = params['model']['name']
    llm_batch_size = params['model']['llm_batch_size']
    str_dtype = params['model']['torch_dtype']
    save_activations = params['activation_caching']['save_activations']

    print(f"\nEvaluation settings:")
    print(f"  Model: {model_name}")
    print(f"  Batch size: {llm_batch_size}")
    print(f"  Save activations: {save_activations}")

    # Temporarily override output folders for SAE Bench
    # This is necessary because run_all_evals_custom_saes expects specific folder names
    old_cwd = os.getcwd()
    try:
        # Create symlinks or set up environment for SAE Bench to find outputs
        for eval_type in eval_types:
            output_dir = output_dirs[eval_type]
            os.makedirs(output_dir, exist_ok=True)

        results = run_all_evals_custom_saes.run_evals(
            model_name,
            selected_saes,
            llm_batch_size,
            str_dtype,
            device,
            eval_types,
            api_key=None,
            force_rerun=False,
            save_activations=save_activations,
        )
    finally:
        os.chdir(old_cwd)

    print(f"\n=== Optional Evaluations Complete ===")

    for eval_type in eval_types:
        output_dir = output_dirs[eval_type]
        print(f"  {eval_type} results saved to: {output_dir}")

    return results


def main(
    saes_dir: str,
    absorption_dir: str,
    scr_dir: str,
    tpp_dir: str,
    unlearning_dir: str
):
    """Run optional evaluations on SAEs

    Args:
        saes_dir: Directory containing SAE weights
        absorption_dir: Directory for absorption eval results
        scr_dir: Directory for SCR eval results
        tpp_dir: Directory for TPP eval results
        unlearning_dir: Directory for unlearning eval results
    """
    # Load parameters
    params = load_params()

    # Get enabled optional evaluation types
    eval_types = get_enabled_eval_types(params)

    # Create output directories even if evals are disabled (DVC expects them)
    for output_dir in [absorption_dir, scr_dir, tpp_dir, unlearning_dir]:
        os.makedirs(output_dir, exist_ok=True)

    if not eval_types:
        print("No optional evaluations enabled in params.yaml")
        print("Created empty output directories for DVC")
        sys.exit(0)

    # Setup device
    from sae_bench.sae_bench_utils.general_utils import setup_environment
    device = setup_environment()
    print(f"Using device: {device}")

    # Load SAEs
    selected_saes = load_saes(saes_dir)

    # Map eval types to output directories
    output_dirs = {
        'absorption': absorption_dir,
        'scr': scr_dir,
        'tpp': tpp_dir,
        'unlearning': unlearning_dir,
    }

    # Run optional evaluations
    run_optional_evaluations(selected_saes, params, device, eval_types, output_dirs)

    print("\n=== Done ===")


if __name__ == "__main__":
    fire.Fire(main)
