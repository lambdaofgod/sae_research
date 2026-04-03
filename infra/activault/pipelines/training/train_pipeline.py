from kfp import dsl, compiler
from kfp import kubernetes


@dsl.component(base_image="sae-training:latest", packages_to_install=[])
def sae_train(
    save_dir: str,
    model_name: str,
    layers: str,
    architectures: str,
    device: str,
    save_checkpoints: bool,
    mlflow: bool,
):
    """Run SAE training via runner.py."""
    import os
    import subprocess

    os.environ["MLFLOW_TRACKING_URI"] = "http://activault-mlflow.default.svc:5000"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://activault-garage.default:3900"

    cmd = [
        "python",
        "-m",
        "sae_research.training.cli_runner",
        "--save_dir",
        save_dir,
        "--model_name",
        model_name,
        "--layers",
        *layers.split(","),
        "--architectures",
        *architectures.split(","),
        "--device",
        device,
    ]
    if save_checkpoints:
        cmd.append("--save_checkpoints")
    if not mlflow:
        cmd.append("--no-mlflow")

    subprocess.run(cmd, check=True)


@dsl.pipeline(name="sae-training")
def training_pipeline(
    save_dir: str = "sae_run",
    model_name: str = "google/gemma-2-2b-it",
    layers: str = "13",
    architectures: str = "batch_top_k",
    device: str = "cuda:0",
    save_checkpoints: bool = False,
    mlflow: bool = True,
):
    task = sae_train(
        save_dir=save_dir,
        model_name=model_name,
        layers=layers,
        architectures=architectures,
        device=device,
        save_checkpoints=save_checkpoints,
        mlflow=mlflow,
    )

    # ── Resources ──────────────────────────────────────────────────────────────
    (
        task.set_memory_request("16Gi")
        .set_memory_limit("32Gi")
        .set_accelerator_type("nvidia.com/gpu")
        .set_accelerator_limit(1)
    )

    # ── S3 credentials ────────────────────────────────────────────────────────
    kubernetes.use_secret_as_env(
        task,
        secret_name="activault-s3-creds",
        secret_key_to_env={
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
            "AWS_ENDPOINT_URL": "AWS_ENDPOINT_URL",
            "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
        },
    )
    kubernetes.use_secret_as_env(
        task,
        secret_name="activault-s3-creds",
        secret_key_to_env={"AWS_ENDPOINT_URL": "S3_ENDPOINT_URL"},
    )

    # ── HuggingFace token ────────────────────────────────────────────────────
    kubernetes.use_secret_as_env(
        task,
        secret_name="hf-token-secret",
        secret_key_to_env={"token": "HF_TOKEN"},
    )

    # ── /dev/shm for PyTorch DataLoader shared memory ────────────────────────
    kubernetes.empty_dir_mount(
        task,
        volume_name="dshm",
        mount_path="/dev/shm",
        medium="Memory",
        size_limit="16Gi",
    )

    # ── Image pull policy (local image in minikube) ──────────────────────────
    kubernetes.set_image_pull_policy(task, "Never")

    # ── Long-run settings ────────────────────────────────────────────────────
    kubernetes.set_timeout(task, 259200)  # 3 days
    task.set_caching_options(False)
    task.set_retry(num_retries=0, backoff_duration="0s")

    # ── Node targeting ───────────────────────────────────────────────────────
    kubernetes.add_toleration(
        task,
        key="nvidia.com/gpu",
        operator="Exists",
        effect="NoSchedule",
    )
    kubernetes.add_pod_annotation(
        task,
        "cluster-autoscaler.kubernetes.io/safe-to-evict",
        "false",
    )


def main(
    output: str,
    submit: bool = False,
    host: str = "http://localhost:8082",
    save_dir: str = "sae_run",
    model_name: str = "google/gemma-2-2b-it",
    layers: str = "13",
    architectures: str = "batch_top_k",
    device: str = "cuda:0",
    save_checkpoints: bool = False,
    mlflow: bool = True,
):
    """Compile and optionally submit SAE training pipeline."""
    compiler.Compiler().compile(training_pipeline, output)
    print(f"Compiled to {output}")

    if submit:
        from kfp.client import Client

        client = Client(host=host)
        run = client.create_run_from_pipeline_func(
            training_pipeline,
            arguments={
                "save_dir": save_dir,
                "model_name": model_name,
                "layers": layers,
                "architectures": architectures,
                "device": device,
                "save_checkpoints": save_checkpoints,
                "mlflow": mlflow,
            },
            experiment_name="sae-training",
            run_name=f"{model_name}_{architectures}",
            enable_caching=False,
        )
        print(f"Run ID: {run.run_id}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
