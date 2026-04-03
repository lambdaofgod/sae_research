"""
Smoke test for the eval reconciler.

Trains a tiny SAE, verifies the reconciler discovers the eval gap,
runs reconciliation to fill it, then verifies no gaps remain.

Usage:
    uv run python smoke_tests/smoke_test_reconciler.py                    # uses K8s MLflow
    uv run python smoke_tests/smoke_test_reconciler.py --backend=local    # uses local server
"""

import argparse
import shutil
import tempfile

import mlflow
import torch as t

from sae_research.training.mlflow_logging import start_sweep_run
from sae_research.training.train import trainSAE
from sae_research.training.config import get_trainer_configs
from sae_research.eval.reconciler import discover_gaps, reconcile

from smoke_tests.smoke_test_mlflow_training import mlflow_backend


MODEL_NAME = "EleutherAI/pythia-70m-deduped"
ACTIVATION_DIM = 512
LAYER = 3
EXPERIMENT_NAME = "smoke_test_reconciler"


def train_tiny_sae(tracking_uri: str, save_dir: str) -> str:
    """Train a tiny SAE and return the child run ID."""
    mlflow.set_tracking_uri(tracking_uri)

    parent_run = start_sweep_run(
        experiment_name=EXPERIMENT_NAME,
        model_name=MODEL_NAME,
        layers=[LAYER],
        architectures=["batch_top_k"],
        run_cfg={"num_tokens": 100},
    )
    parent_run_id = parent_run.info.run_id

    configs = get_trainer_configs(
        architectures=["batch_top_k"],
        learning_rates=[1e-4],
        seeds=[0],
        activation_dim=ACTIVATION_DIM,
        dict_sizes=[1024],
        model_name=MODEL_NAME,
        device="cpu",
        layer=LAYER,
        submodule_name=f"resid_post_layer_{LAYER}",
        steps=100,
        warmup_steps=1,
        sparsity_warmup_steps=1,
        decay_start_fraction=0.8,
        anneal_end_fraction=0.01,
    )
    configs = [configs[0]]

    def dummy_data():
        while True:
            yield t.randn(32, ACTIVATION_DIM)

    run_ids = trainSAE(
        data=dummy_data(),
        trainer_configs=configs,
        steps=5,
        use_mlflow=True,
        mlflow_parent_run_id=parent_run_id,
        save_dir=save_dir,
        log_steps=1,
        device="cpu",
    )
    mlflow.end_run()

    assert len(run_ids) == 1
    return run_ids[0]


def run_smoke_test(tracking_uri: str):
    save_dir = tempfile.mkdtemp(prefix="reconciler_smoke_saes_")

    try:
        # --- Step 1: Train a tiny SAE ---
        print("\n=== Step 1: Training tiny SAE ===")
        child_run_id = train_tiny_sae(tracking_uri, save_dir)
        print(f"Training done. Child run: {child_run_id}")

        # --- Step 2: Discover gaps (should find core eval missing) ---
        print("\n=== Step 2: Discovering gaps ===")
        gaps = discover_gaps(
            tracking_uri=tracking_uri,
            experiment_name=EXPERIMENT_NAME,
            eval_types=["core"],
        )
        print(f"Found {len(gaps)} gaps")
        assert len(gaps) >= 1, f"Expected at least 1 gap, got {len(gaps)}"

        our_gap = [g for g in gaps if g["run_id"] == child_run_id]
        assert len(our_gap) == 1, (
            f"Expected gap for run {child_run_id}, not found in {gaps}"
        )
        assert our_gap[0]["eval_type"] == "core"
        print(f"PASS: gap found for run {child_run_id[:8]}, eval_type=core")

        # --- Step 3: Run reconciliation ---
        print("\n=== Step 3: Running reconciliation ===")
        results = reconcile(
            tracking_uri=tracking_uri,
            experiment_name=EXPERIMENT_NAME,
            eval_types=["core"],
            run_ids=[child_run_id],
            device="cpu",
            dtype="float32",
            eval_config={
                "n_reconstruction_batches": 1,
                "n_sparsity_batches": 1,
                "batch_size": 1,
                "context_size": 64,
            },
        )
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        assert len(results[0]["metrics"]) > 0, "No metrics returned"
        print(f"PASS: reconciliation produced {len(results[0]['metrics'])} metrics")

        # --- Step 4: Verify no gaps remain ---
        print("\n=== Step 4: Verifying no gaps remain ===")
        gaps_after = discover_gaps(
            tracking_uri=tracking_uri,
            experiment_name=EXPERIMENT_NAME,
            eval_types=["core"],
            run_ids=[child_run_id],
        )
        assert len(gaps_after) == 0, (
            f"Expected 0 gaps after reconciliation, got {len(gaps_after)}"
        )
        print("PASS: no gaps remain")

        # --- Step 5: Verify --force re-discovers the gap ---
        print("\n=== Step 5: Verifying --force flag ===")
        gaps_forced = discover_gaps(
            tracking_uri=tracking_uri,
            experiment_name=EXPERIMENT_NAME,
            eval_types=["core"],
            run_ids=[child_run_id],
            force=True,
        )
        assert len(gaps_forced) == 1, (
            f"Expected 1 gap with force=True, got {len(gaps_forced)}"
        )
        print("PASS: force=True re-discovers the gap")

        # --- Step 6: Verify metrics in MLflow ---
        print("\n=== Step 6: Verifying metrics in MLflow ===")
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient()
        run = client.get_run(child_run_id)

        for key in ["core_l0", "core_explained_variance"]:
            assert key in run.data.metrics, (
                f"Metric {key} not found. Got: {list(run.data.metrics.keys())}"
            )
            print(f"PASS: {key} = {run.data.metrics[key]}")

        assert run.data.tags.get("experiment_name") == EXPERIMENT_NAME
        assert run.data.tags.get("eval_type") == "core"
        print("PASS: tags correct")

        print("\nAll reconciler smoke tests passed!")

    finally:
        shutil.rmtree(save_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Eval reconciler smoke test")
    parser.add_argument(
        "--backend",
        choices=["kubernetes", "local"],
        default="kubernetes",
        help="MLflow backend to test against (default: kubernetes)",
    )
    args = parser.parse_args()

    with mlflow_backend(args.backend) as tracking_uri:
        run_smoke_test(tracking_uri)


if __name__ == "__main__":
    main()
