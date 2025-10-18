"""
Run Sparse Probing Evaluation: Tests SAE features for downstream tasks
"""
import os
import sys
import yaml
import pickle
import torch
import fire
from pathlib import Path

import sae_bench.evals.sparse_probing.main as sparse_probing
import sae_bench.sae_bench_utils.general_utils as general_utils


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


def run_sparse_probing_evaluation(selected_saes, params, device, output_dir: str):
    """Run sparse probing evaluation on selected SAEs"""
    print("\n=== Running Sparse Probing Evaluation ===")

    model_name = params['model']['name']
    random_seed = params['random_seed']
    llm_batch_size = params['model']['llm_batch_size']
    str_dtype = params['model']['torch_dtype']
    dataset_names = params['sparse_probing']['dataset_names']
    save_activations = params['activation_caching']['save_activations']

    print(f"Evaluation settings:")
    print(f"  Model: {model_name}")
    print(f"  Batch size: {llm_batch_size}")
    print(f"  Datasets: {dataset_names}")
    print(f"  Save activations: {save_activations}")

    eval_config = sparse_probing.SparseProbingEvalConfig(
        model_name=model_name,
        random_seed=random_seed,
        llm_batch_size=llm_batch_size,
        llm_dtype=str_dtype,
        dataset_names=dataset_names,
        sae_batch_size=params['sparse_probing']['sae_batch_size'],
    )

    results = sparse_probing.run_eval(
        eval_config,
        selected_saes,
        device,
        output_dir,
        force_rerun=False,
        clean_up_activations=True,
        save_activations=save_activations,
    )

    print(f"\n=== Sparse Probing Evaluation Complete ===")
    print(f"Results saved to: {output_dir}")

    return results


def main(saes_dir: str, output_dir: str, gpu_id: int = None):
    """Run sparse probing evaluation on SAEs

    Args:
        saes_dir: Directory containing SAE weights
        output_dir: Directory to save evaluation results
        gpu_id: GPU ID to use (if None, reads from params.yaml or uses default)
    """
    # Load parameters
    params = load_params()

    # Check if sparse probing evaluation is enabled
    if not params['eval_types']['sparse_probing']:
        print("Sparse probing evaluation is disabled in params.yaml")
        sys.exit(0)

    # Setup GPU
    if gpu_id is None:
        gpu_id = params.get('gpu_assignment', {}).get('sparse_probing_eval', 0)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"Using GPU {gpu_id} for sparse_probing (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")

    # Setup device
    device = general_utils.setup_environment()
    print(f"Using device: {device}")

    # Load SAEs
    selected_saes = load_saes(saes_dir)

    # Run sparse probing evaluation
    run_sparse_probing_evaluation(selected_saes, params, device, output_dir)

    print("\n=== Done ===")


if __name__ == "__main__":
    fire.Fire(main)
