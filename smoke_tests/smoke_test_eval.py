"""
Smoke test for SAE evaluation pipeline.

Runs the training smoke test first to produce an SAE in MLflow, then pulls
the SAE artifacts, runs sae_bench core eval with minimal settings, and
logs eval metrics back to the same MLflow run.

Usage:
    uv run python smoke_tests/smoke_test_eval.py                    # uses K8s MLflow
    uv run python smoke_tests/smoke_test_eval.py --backend=local    # uses local server
"""

import shutil
import tempfile

import mlflow
import torch as t

from sae_research.training.train import trainSAE
from sae_research.training.config import get_trainer_configs
from sae_research.eval.sae_wrapper import load_sae_for_eval
from sae_research.eval.components import (
    run_core_eval,
    extract_core_metrics,
    log_eval_to_mlflow,
)

# Reuse the MLflow backend context manager from the training smoke test
from smoke_tests.smoke_test_training import mlflow_backend


# Must match pythia-70m hidden_size
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
ACTIVATION_DIM = 512
LAYER = 3
EXPERIMENT_NAME = "smoke_test_eval"


def train_tiny_sae(save_dir: str) -> str:
    """Train a tiny SAE and return the MLflow run ID."""
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

    mlflow_run_ids = trainSAE(
        data=dummy_data(),
        trainer_configs=configs,
        steps=5,
        mlflow_experiment=EXPERIMENT_NAME,
        save_dir=save_dir,
        log_steps=1,
        device="cpu",
    )

    assert len(mlflow_run_ids) == 1
    return mlflow_run_ids[0]


def run_smoke_test(tracking_uri: str):
    save_dir = tempfile.mkdtemp(prefix="eval_smoke_saes_")
    eval_output_dir = tempfile.mkdtemp(prefix="eval_smoke_results_")

    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient()

        # --- Step 1: Train a tiny SAE ---
        print("\n=== Step 1: Training tiny SAE ===")
        run_id = train_tiny_sae(save_dir)
        print(f"Training done. Run: {run_id}")

        # --- Step 2: Pull SAE artifacts from MLflow ---
        print("\n=== Step 2: Pulling SAE from MLflow ===")
        artifact_dir = tempfile.mkdtemp(prefix="eval_smoke_artifacts_")
        for artifact_name in ["ae.pt", "config.json"]:
            local_path = client.download_artifacts(run_id, artifact_name, artifact_dir)
            print(f"Downloaded: {local_path}")

        # --- Step 3: Load SAE and wrap for sae_bench ---
        print("\n=== Step 3: Loading SAE for eval ===")
        sae_name, wrapped_sae = load_sae_for_eval(
            sae_path=artifact_dir,
            model_name=MODEL_NAME,
            hook_layer=LAYER,
            device="cpu",
            dtype=t.float32,
        )
        print(
            f"Loaded SAE: {sae_name}, d_in={wrapped_sae.cfg.d_in}, d_sae={wrapped_sae.cfg.d_sae}"
        )
        print(f"Architecture: {wrapped_sae.cfg.architecture}")

        # --- Step 4: Run core eval with minimal settings ---
        print("\n=== Step 4: Running core eval (minimal) ===")
        results = run_core_eval(
            selected_saes=[(sae_name, wrapped_sae)],
            n_reconstruction_batches=1,
            n_sparsity_batches=1,
            batch_size=1,
            context_size=64,
            dtype="float32",
            device="cpu",
            output_folder=eval_output_dir,
        )
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        print(f"Core eval complete. Raw result keys: {list(results[0].keys())}")

        # --- Step 5: Extract and log metrics to MLflow ---
        print("\n=== Step 5: Extracting and logging metrics ===")
        metrics = extract_core_metrics(results[0])
        print(f"Extracted metrics: {metrics}")
        assert len(metrics) > 0, "No metrics extracted"

        log_eval_to_mlflow(
            run_id=run_id,
            metrics=metrics,
            tags={
                "experiment_name": "smoke_test",
                "eval_type": "core",
            },
        )
        print(f"Logged {len(metrics)} metrics to run {run_id}")

        # --- Step 6: Verify metrics and tags in MLflow ---
        print("\n=== Step 6: Verification ===")
        run = client.get_run(run_id)
        for key in ["core_l0", "core_explained_variance"]:
            assert key in run.data.metrics, (
                f"Expected metric {key} not found. Got: {list(run.data.metrics.keys())}"
            )
            print(f"PASS: {key} = {run.data.metrics[key]}")

        assert run.data.tags.get("experiment_name") == "smoke_test"
        print(f"PASS: experiment_name tag = {run.data.tags['experiment_name']}")

        print("\nAll eval smoke tests passed!")

    finally:
        shutil.rmtree(save_dir, ignore_errors=True)
        shutil.rmtree(eval_output_dir, ignore_errors=True)
        if "artifact_dir" in locals():
            shutil.rmtree(artifact_dir, ignore_errors=True)


def main():
    """MLflow eval smoke test."""
    with mlflow_backend() as tracking_uri:
        run_smoke_test(tracking_uri)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
