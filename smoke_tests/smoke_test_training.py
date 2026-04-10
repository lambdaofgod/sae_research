"""
Smoke test for SAE training.

Tests both standard (batch_top_k) and temporal (temporal_batch_top_k) architectures
via a stub activation buffer that exercises the real ActivaultS3ActivationBuffer
__next__() batching logic without needing S3/Activault.

Usage:
    uv run python smoke_tests/smoke_test_training.py
"""

import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from contextlib import contextmanager

import mlflow
import torch as t

from dictionary_learning.activault_s3_buffer import ActivaultS3ActivationBuffer

from sae_research.training.mlflow_logging import start_sweep_run
from sae_research.training.train import trainSAE
from sae_research.training.config import get_trainer_configs


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_for_server(url, timeout=30):
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{url}/health")
            return True
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(0.5)
    return False


@contextmanager
def mlflow_backend():
    """Spin up a temporary local MLflow server for smoke tests."""
    tmpdir = tempfile.mkdtemp(prefix="mlflow_smoke_")
    port = find_free_port()
    tracking_uri = f"http://127.0.0.1:{port}"

    server_proc = subprocess.Popen(
        [
            "mlflow",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--backend-store-uri",
            f"sqlite:///{tmpdir}/mlflow.db",
            "--default-artifact-root",
            os.path.join(tmpdir, "artifacts"),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )

    try:
        assert wait_for_server(tracking_uri), "Local MLflow server did not start"
        print(f"Local MLflow server running on {tracking_uri}")
        yield tracking_uri
    finally:
        os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
        server_proc.wait(timeout=5)
        shutil.rmtree(tmpdir, ignore_errors=True)


class StubActivationBuffer(ActivaultS3ActivationBuffer):
    """Stub buffer that fills self.states with synthetic data.

    Exercises the real __next__() batching/read_mask logic from the parent
    class without needing S3 or Activault.
    """

    def __init__(
        self,
        activation_dim: int,
        batch_size: int,
        temporal: bool = False,
        device: str = "cpu",
    ):
        # Skip parent __init__ (needs S3RCache) — set up state directly
        self.batch_size = batch_size
        self.device = device
        self.temporal = temporal
        self.activation_dim = activation_dim
        self.states = None
        self.read_mask = None
        self.refresh()

    def refresh(self):
        n = self.batch_size * 4
        if self.temporal:
            self.states = t.randn(n, 2, self.activation_dim, device=self.device)
        else:
            self.states = t.randn(n, self.activation_dim, device=self.device)
        self.read_mask = t.zeros(n, dtype=t.bool, device=self.device)


MODEL_NAME = "EleutherAI/pythia-70m-deduped"
ACTIVATION_DIM = 512
LAYER = 3


def run_single_architecture(
    tracking_uri: str, architecture: str, expected_trainer_class: str
):
    """Train one architecture and verify MLflow logging."""
    save_dir = tempfile.mkdtemp(prefix=f"smoke_{architecture}_")
    temporal = "temporal" in architecture

    try:
        print(f"\n{'=' * 60}")
        print(f"Testing architecture: {architecture}")
        print(f"{'=' * 60}")

        parent_run = start_sweep_run(
            experiment_name=f"smoke_test_{architecture}",
            model_name=MODEL_NAME,
            layers=[LAYER],
            architectures=[architecture],
            run_cfg={"num_tokens": 100},
        )
        parent_run_id = parent_run.info.run_id

        configs = get_trainer_configs(
            architectures=[architecture],
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

        stub_buffer = StubActivationBuffer(
            activation_dim=ACTIVATION_DIM,
            batch_size=32,
            temporal=temporal,
            device="cpu",
        )

        run_ids = trainSAE(
            data=stub_buffer,
            trainer_configs=configs,
            steps=5,
            use_mlflow=True,
            mlflow_parent_run_id=parent_run_id,
            save_dir=save_dir,
            log_steps=1,
            device="cpu",
        )
        mlflow.end_run()

        # --- Assertions ---
        assert len(run_ids) == 1, f"Expected 1 run ID, got {len(run_ids)}"
        child_run_id = run_ids[0]

        client = mlflow.MlflowClient()

        child = client.get_run(child_run_id)
        assert child.data.tags.get("mlflow.parentRunId") == parent_run_id
        print("PASS: child run is nested under parent")

        assert child.data.params.get("trainer_class") == expected_trainer_class
        print(f"PASS: trainer_class={expected_trainer_class}")

        metric_history = client.get_metric_history(child_run_id, "l0")
        assert len(metric_history) > 0, "No l0 metrics logged"
        print(f"PASS: metrics logged ({len(metric_history)} l0 data points)")

        artifacts = client.list_artifacts(child_run_id)
        artifact_names = [a.path for a in artifacts]
        assert "ae.pt" in artifact_names, f"ae.pt not in artifacts: {artifact_names}"
        print(f"PASS: artifacts logged ({artifact_names})")

    finally:
        shutil.rmtree(save_dir, ignore_errors=True)


def run_smoke_test(tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)

    run_single_architecture(tracking_uri, "batch_top_k", "BatchTopKTrainer")
    run_single_architecture(
        tracking_uri, "temporal_batch_top_k", "TemporalBatchTopKTrainer"
    )

    print("\nAll training smoke tests passed!")


def main():
    """SAE training smoke test."""
    with mlflow_backend() as tracking_uri:
        run_smoke_test(tracking_uri)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
