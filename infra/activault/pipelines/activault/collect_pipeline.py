from kfp import dsl, compiler
from kfp import kubernetes


@dsl.component(base_image="activault-flashattn:latest", packages_to_install=[])
def activault_collect(
    run_name: str,
    model_name: str,
    dtype: str,
    n_batches: int,
    seq_length: int,
    batch_size: int,
    seed: int,
    start_batch: int,
    batches_per_upload: int,
    hooks: str,
    bucket_name: str,
    dataset: str,
):
    """Generate config.yaml from parameters and run activault collect."""
    import subprocess
    import yaml

    cfg = {
        "run_name": run_name,
        "num_runs": 1,
        "transformer_config": {
            "model_name": model_name,
            "dtype": dtype,
            "cache_dir": "/cache",
            "max_per_device_memory": "24GB",
        },
        "data_config": {
            "bucket_name": bucket_name,
            "data_key": dataset,
            "n_batches": n_batches,
            "seq_length": seq_length,
            "batch_size": batch_size,
            "seed": seed,
            "skip_cache": False,
            "start_batch": start_batch,
            "clean_added_tokens": False,
            "clean_default_system_prompt": False,
        },
        "upload_config": {
            "batches_per_upload": batches_per_upload,
            "hooks": hooks.split(","),
        },
    }

    with open("/tmp/config.yaml", "w") as f:
        yaml.dump(cfg, f)

    subprocess.run(
        ["activault", "collect", "--config", "/tmp/config.yaml"],
        check=True,
    )


@dsl.pipeline(name="activault-collect")
def activault_pipeline(
    run_name: str = "gemma2_2b",
    model_name: str = "google/gemma-2-2b-it",
    dtype: str = "bfloat16",
    n_batches: int = 3600,
    seq_length: int = 256,
    batch_size: int = 64,
    seed: int = 42,
    start_batch: int = 0,
    batches_per_upload: int = 4,
    hooks: str = "models.layers.13.self_attn.post,models.layers.13.mlp.post",
    bucket_name: str = "activations",
    dataset: str = "monology/pile-uncopyrighted",
):
    task = activault_collect(
        run_name=run_name,
        model_name=model_name,
        dtype=dtype,
        n_batches=n_batches,
        seq_length=seq_length,
        batch_size=batch_size,
        seed=seed,
        start_batch=start_batch,
        batches_per_upload=batches_per_upload,
        hooks=hooks,
        bucket_name=bucket_name,
        dataset=dataset,
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
    config: str = "",
    host: str = "http://localhost:8081",
):
    """Compile and optionally submit activault collection pipeline."""
    import yaml

    compiler.Compiler().compile(activault_pipeline, output)
    print(f"Compiled to {output}")

    if submit:
        if not config:
            raise ValueError("--config is required when using --submit")
        with open(config) as f:
            cfg = yaml.safe_load(f)

        arguments = {
            "run_name": cfg["run_name"],
            "model_name": cfg["transformer_config"]["model_name"],
            "dtype": cfg["transformer_config"]["dtype"],
            "n_batches": cfg["data_config"]["n_batches"],
            "seq_length": cfg["data_config"]["seq_length"],
            "batch_size": cfg["data_config"]["batch_size"],
            "seed": cfg["data_config"]["seed"],
            "start_batch": cfg["data_config"]["start_batch"],
            "batches_per_upload": cfg["upload_config"]["batches_per_upload"],
            "hooks": ",".join(cfg["upload_config"]["hooks"]),
            "bucket_name": cfg["data_config"]["bucket_name"],
            "dataset": cfg["data_config"]["data_key"],
        }

        from kfp.client import Client

        client = Client(host=host)
        run = client.create_run_from_pipeline_func(
            activault_pipeline,
            arguments=arguments,
            experiment_name="activault",
            run_name=cfg["run_name"],
            enable_caching=False,
        )
        print(f"Run ID: {run.run_id}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
