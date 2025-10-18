"""
Run a Single Optional Evaluation: Absorption, SCR, TPP, or Unlearning

Usage:
    python run_optional_evals.py <eval_type> <saes_dir> <output_dir>

Where eval_type is one of: scr, tpp, absorption, unlearning

The script checks params.yaml to see if the evaluation is enabled.
If disabled, it creates an empty output directory and exits.
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


# Removed get_enabled_eval_types - no longer needed since we run one eval type at a time


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
    eval_type: str,
    saes_dir: str,
    output_dir: str
):
    """Run a single optional evaluation type on SAEs

    Args:
        eval_type: Evaluation type to run (one of: 'scr', 'tpp', 'absorption', 'unlearning')
        saes_dir: Directory containing SAE weights
        output_dir: Directory for this evaluation's results
    """
    # Load parameters
    params = load_params()

    # Validate eval_type
    valid_eval_types = ['absorption', 'scr', 'tpp', 'unlearning']
    if eval_type not in valid_eval_types:
        print(f"Error: eval_type must be one of {valid_eval_types}, got: {eval_type}")
        sys.exit(1)

    # Check if this eval type is enabled in config
    eval_enabled = params['eval_types'].get(eval_type, False)
    if not eval_enabled:
        print(f"{eval_type} evaluation is disabled in params.yaml")
        print(f"Creating empty output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        sys.exit(0)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup device
    from sae_bench.sae_bench_utils.general_utils import setup_environment
    device = setup_environment()
    print(f"Using device: {device}")

    # Load SAEs
    selected_saes = load_saes(saes_dir)

    # Run this single evaluation type
    output_dirs = {eval_type: output_dir}
    run_optional_evaluations(selected_saes, params, device, [eval_type], output_dirs)

    print("\n=== Done ===")


if __name__ == "__main__":
    fire.Fire(main)
