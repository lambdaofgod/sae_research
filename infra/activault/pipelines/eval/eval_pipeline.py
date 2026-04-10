"""
KFP v2 pipeline for SAE evaluation with reconciler pattern.

Discovers which (SAE, eval_type) pairs are missing in MLflow, runs evaluations
sequentially on a single GPU, and logs results back to each SAE's MLflow run.

Usage:
    # Compile only
    python eval_pipeline.py --compile-only

    # Submit to KFP (full reconciliation)
    python eval_pipeline.py --submit --config config.yaml

    # Submit targeted rerun
    python eval_pipeline.py --submit \
        --config config.yaml \
        --override experiment_name=my_experiment \
        --override eval_types=core \
        --override force=true
"""

from kfp import compiler, dsl
from kfp import kubernetes


@dsl.component(base_image="sae-eval:latest", packages_to_install=[])
def reconcile_and_eval(
    tracking_uri: str,
    experiment_name: str,
    eval_types: str,
    run_ids: str,
    force: bool,
    device: str,
    dtype: str,
) -> str:
    """Discover eval gaps and run evaluations sequentially.

    All imports are inside the function body because KFP serializes this
    function and executes it inside the container image.
    """
    import json

    from eval_reconciler import reconcile

    if not experiment_name:
        raise ValueError(
            "experiment_name is required. Example: 'topk_vs_batchtopk_march2024'"
        )

    eval_type_list = [t.strip() for t in eval_types.split(",") if t.strip()]
    run_id_list = [r.strip() for r in run_ids.split(",") if r.strip()] or None

    results = reconcile(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        eval_types=eval_type_list if eval_type_list else None,
        run_ids=run_id_list,
        force=force,
        device=device,
        dtype=dtype,
    )

    return json.dumps(results, default=str)


@dsl.pipeline(name="sae-eval-reconcile")
def eval_pipeline(
    tracking_uri: str = "http://activault-mlflow.default.svc:5000",
    experiment_name: str = "",
    eval_types: str = "core",
    run_ids: str = "",
    force: bool = False,
    device: str = "cuda",
    dtype: str = "float32",
):
    """SAE evaluation reconciler pipeline.

    Discovers missing evaluations and runs them sequentially.

    Args:
        tracking_uri: MLflow tracking server URI (in-cluster).
        experiment_name: REQUIRED. Tag for eval campaign isolation.
        eval_types: Comma-separated eval types (e.g. "core" or "core,sparse_probing").
        run_ids: Comma-separated MLflow run IDs. Empty = reconcile all SAE runs.
        force: Rerun even if metrics already exist.
        device: Eval device (cuda, cpu).
        dtype: Eval dtype (float32, bfloat16).
    """
    task = reconcile_and_eval(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        eval_types=eval_types,
        run_ids=run_ids,
        force=force,
        device=device,
        dtype=dtype,
    )

    # ── Resources ──────────────────────────────────────────────────────────────
    (
        task.set_memory_request("16Gi")
        .set_memory_limit("32Gi")
        .set_accelerator_type("nvidia.com/gpu")
        .set_accelerator_limit(1)
    )

    # ── S3 credentials (shared with training — Garage backend) ─────────────────
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

    # ── MLflow S3 endpoint (for artifact downloads) ────────────────────────────
    kubernetes.use_secret_as_env(
        task,
        secret_name="activault-s3-creds",
        secret_key_to_env={"AWS_ENDPOINT_URL": "MLFLOW_S3_ENDPOINT_URL"},
    )

    # ── HuggingFace token (model + dataset downloads) ──────────────────────────
    kubernetes.use_secret_as_env(
        task,
        secret_name="hf-token-secret",
        secret_key_to_env={"token": "HF_TOKEN"},
    )

    # ── /dev/shm for PyTorch DataLoader shared memory ──────────────────────────
    kubernetes.empty_dir_mount(
        task,
        volume_name="dshm",
        mount_path="/dev/shm",
        medium="Memory",
        size_limit="16Gi",
    )

    # ── Image pull policy (local image in minikube) ────────────────────────────
    kubernetes.set_image_pull_policy(task, "Never")

    # ── Long-run settings ──────────────────────────────────────────────────────
    kubernetes.set_timeout(task, 259200)  # 3 days
    task.set_caching_options(False)
    task.set_retry(num_retries=0, backoff_duration="0s")

    # ── GPU scheduling ─────────────────────────────────────────────────────────
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
    host: str = "http://localhost:8087",
    **overrides,
):
    """Compile and optionally submit SAE eval reconciler pipeline.

    Extra keyword arguments override config values, e.g.:
        --experiment_name=my_experiment --force=true
    """
    import yaml

    compiler.Compiler().compile(eval_pipeline, output)
    print(f"Compiled pipeline to {output}")

    if submit:
        if not config:
            raise ValueError("--config is required when using --submit")
        with open(config) as f:
            cfg = yaml.safe_load(f)

        for key, value in overrides.items():
            cfg[key] = value

        if not cfg.get("experiment_name"):
            raise ValueError(
                "experiment_name is required (set in config.yaml or via override)"
            )

        from kfp.client import Client

        client = Client(host=host)
        run = client.create_run_from_pipeline_func(
            eval_pipeline,
            arguments=cfg,
            experiment_name="sae-eval",
            run_name=f"eval_{cfg['experiment_name']}",
            enable_caching=False,
        )
        print(f"Run ID: {run.run_id}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
