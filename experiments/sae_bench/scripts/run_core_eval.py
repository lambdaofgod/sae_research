"""
Run Core Evaluation: Basic reconstruction and sparsity metrics
"""
import os
import sys
import yaml
import pickle
import fire
from pathlib import Path

import sae_bench.evals.core.main as core


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


def run_core_evaluation(selected_saes, params, output_dir: str):
    """Run core evaluation on selected SAEs"""
    print("\n=== Running Core Evaluation ===")

    core_params = params['core_eval']
    str_dtype = params['model']['torch_dtype']

    print(f"Evaluation settings:")
    print(f"  Reconstruction batches: {core_params['n_eval_reconstruction_batches']}")
    print(f"  Sparsity/variance batches: {core_params['n_eval_sparsity_variance_batches']}")
    print(f"  Batch size: {core_params['eval_batch_size_prompts']}")
    print(f"  Dataset: {core_params['dataset']}")
    print(f"  Context size: {core_params['context_size']}")

    results = core.multiple_evals(
        selected_saes=selected_saes,
        n_eval_reconstruction_batches=core_params['n_eval_reconstruction_batches'],
        n_eval_sparsity_variance_batches=core_params['n_eval_sparsity_variance_batches'],
        eval_batch_size_prompts=core_params['eval_batch_size_prompts'],
        compute_featurewise_density_statistics=core_params['compute_featurewise_density_statistics'],
        compute_featurewise_weight_based_metrics=core_params['compute_featurewise_weight_based_metrics'],
        exclude_special_tokens_from_reconstruction=core_params['exclude_special_tokens_from_reconstruction'],
        dataset=core_params['dataset'],
        context_size=core_params['context_size'],
        output_folder=output_dir,
        verbose=True,
        dtype=str_dtype,
    )

    print(f"\n=== Core Evaluation Complete ===")
    print(f"Results saved to: {output_dir}")

    return results


def main(saes_dir: str, output_dir: str):
    """Run core evaluation on SAEs

    Args:
        saes_dir: Directory containing SAE weights
        output_dir: Directory to save evaluation results
    """
    # Load parameters
    params = load_params()

    # Check if core evaluation is enabled
    if not params['eval_types']['core']:
        print("Core evaluation is disabled in params.yaml")
        sys.exit(0)

    # Load SAEs
    selected_saes = load_saes(saes_dir)

    # Run core evaluation
    run_core_evaluation(selected_saes, params, output_dir)

    print("\n=== Done ===")


if __name__ == "__main__":
    fire.Fire(main)
