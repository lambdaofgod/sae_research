"""
Smoke test for MLflow integration with SAE training.

By default, connects to the K8s MLflow server (via proxy at localhost:8085).
Pass --backend=local to spin up a temporary file-backed server instead.

Usage:
    uv run python test/test_mlflow_training_smoke.py                    # uses K8s MLflow
    uv run python test/test_mlflow_training_smoke.py --backend=local    # uses local server
"""

import argparse
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

from sae_research.training.mlflow_logging import start_sweep_run
from sae_research.training.train import trainSAE
from sae_research.training.config import get_trainer_configs


K8S_MLFLOW_URI = "http://localhost:8085"


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
def mlflow_backend(backend: str):
    """Provide an MLflow tracking URI based on the chosen backend."""
    if backend == "kubernetes":
        assert wait_for_server(K8S_MLFLOW_URI, timeout=5), (
            f"K8s MLflow not reachable at {K8S_MLFLOW_URI}. "
            f"Is the port-forward running? "
            f"(kubectl port-forward svc/activault-proxy 8085:8085)"
        )
        print(f"Using K8s MLflow at {K8S_MLFLOW_URI}")
        yield K8S_MLFLOW_URI

    elif backend == "local":
        tmpdir = tempfile.mkdtemp(prefix="mlflow_smoke_")
        port = find_free_port()
        tracking_uri = f"http://127.0.0.1:{port}"

        server_proc = subprocess.Popen(
            [
                "mlflow", "server",
                "--host", "127.0.0.1",
                "--port", str(port),
                "--backend-store-uri", f"sqlite:///{tmpdir}/mlflow.db",
                "--default-artifact-root", os.path.join(tmpdir, "artifacts"),
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

    else:
        raise ValueError(f"Unknown backend: {backend}")


def run_smoke_test(tracking_uri: str):
    save_dir = tempfile.mkdtemp(prefix="mlflow_smoke_saes_")

    try:
        mlflow.set_tracking_uri(tracking_uri)

        # Create parent sweep run
        parent_run = start_sweep_run(
            experiment_name="smoke_test",
            model_name="test_model",
            layers=[0],
            architectures=["batch_top_k"],
            run_cfg={"num_tokens": 100},
        )
        parent_run_id = parent_run.info.run_id
        print(f"Parent run: {parent_run_id}")

        # Build a minimal trainer config
        configs = get_trainer_configs(
            architectures=["batch_top_k"],
            learning_rates=[1e-4],
            seeds=[0],
            activation_dim=64,
            dict_sizes=[128],
            model_name="test_model",
            device="cpu",
            layer=0,
            submodule_name="test",
            steps=100,
            warmup_steps=1,
            sparsity_warmup_steps=1,
            decay_start_fraction=0.8,
            anneal_end_fraction=0.01,
        )
        assert len(configs) > 0, "No trainer configs generated"
        configs = [configs[0]]

        def dummy_data():
            while True:
                yield t.randn(32, 64)

        mlflow_run_ids = trainSAE(
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

        # --- Assertions ---
        assert len(mlflow_run_ids) == 1, f"Expected 1 child run, got {len(mlflow_run_ids)}"
        child_run_id = mlflow_run_ids[0]

        client = mlflow.MlflowClient()

        parent = client.get_run(parent_run_id)
        assert parent.info.run_id == parent_run_id
        print(f"PASS: parent run exists (tags: {parent.data.tags.get('sweep')})")

        child = client.get_run(child_run_id)
        assert child.data.tags.get("mlflow.parentRunId") == parent_run_id
        print(f"PASS: child run is nested under parent")

        assert "dict_size" in child.data.params
        assert "trainer_class" in child.data.params
        print(f"PASS: params logged (dict_size={child.data.params['dict_size']})")

        metric_history = client.get_metric_history(child_run_id, "l0")
        assert len(metric_history) > 0, "No l0 metrics logged"
        print(f"PASS: metrics logged ({len(metric_history)} l0 data points)")

        artifacts = client.list_artifacts(child_run_id)
        artifact_names = [a.path for a in artifacts]
        assert "ae.pt" in artifact_names, f"ae.pt not in artifacts: {artifact_names}"
        assert "config.json" in artifact_names, f"config.json not in artifacts: {artifact_names}"
        print(f"PASS: artifacts logged ({artifact_names})")

        print("\nAll smoke tests passed!")

    finally:
        shutil.rmtree(save_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="MLflow training smoke test")
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
