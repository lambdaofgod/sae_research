"""
MLflow logging helpers for SAE training.

Provides a thin wrapper around mlflow to handle parent/child run hierarchy
and standardized param/metric logging for SAE sweeps.
"""

import os

import mlflow
import torch as t


def start_sweep_run(
    experiment_name: str,
    model_name: str,
    layers: list[int],
    architectures: list[str],
    run_cfg: dict,
) -> mlflow.ActiveRun:
    """Start a parent MLflow run for a sweep (one runner.py invocation)."""
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(
        run_name=f"{model_name}_{'_'.join(architectures)}",
        tags={
            "sweep": "true",
            "model_name": model_name,
            "layers": str(layers),
            "architectures": ",".join(architectures),
        },
    )
    # Log sweep-level params
    safe_cfg = {k: v for k, v in run_cfg.items() if not isinstance(v, (t.Tensor, type))}
    mlflow.log_params(safe_cfg)
    return run


def start_trainer_run(
    parent_run_id: str | None,
    trainer_index: int,
    trainer_config: dict,
) -> mlflow.entities.Run:
    """Start a child MLflow run for one trainer within a sweep.

    Returns an mlflow.entities.Run (not ActiveRun) — the run stays in
    RUNNING status until end_trainer_run() is called.
    """
    client = _get_client()

    param_keys = [
        "activation_dim",
        "dict_size",
        "seed",
        "lr",
        "layer",
        "lm_name",
        "submodule_name",
        "steps",
        "warmup_steps",
        "decay_start",
        # Architecture-specific
        "k",
        "l1_penalty",
        "target_l0",
        "initial_sparsity_penalty",
        "sparsity_warmup_steps",
        "k_values",
        "s",
        "auxk_alpha",
        "threshold_beta",
    ]

    trainer_path = str(trainer_config.get("trainer", ""))
    trainer_class_name = trainer_path.rsplit(".", 1)[-1] if trainer_path else ""

    dict_class_path = str(trainer_config.get("dict_class", ""))
    dict_class_name = dict_class_path.rsplit(".", 1)[-1] if dict_class_path else ""

    run_name = f"trainer_{trainer_index}_{trainer_class_name}"

    experiment_id = "0"
    if parent_run_id:
        parent_run = client.get_run(parent_run_id)
        experiment_id = parent_run.info.experiment_id

    run = client.create_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags={
            "trainer_index": str(trainer_index),
            "trainer_class": trainer_class_name,
            "dict_class": dict_class_name,
            "mlflow.parentRunId": parent_run_id or "",
        },
    )

    params = {}
    for key in param_keys:
        if key in trainer_config:
            val = trainer_config[key]
            if isinstance(val, t.Tensor):
                val = val.item()
            elif isinstance(val, list):
                val = str(val)
            params[key] = val

    params["trainer_class"] = trainer_class_name
    params["dict_class"] = dict_class_name

    for key, val in params.items():
        client.log_param(run.info.run_id, key, val)

    return run


def end_trainer_run(run_id: str):
    """Mark a trainer run as FINISHED."""
    _get_client().set_terminated(run_id)


_client = None


def _get_client() -> mlflow.tracking.MlflowClient:
    global _client
    if _client is None:
        _client = mlflow.tracking.MlflowClient()
    return _client


def log_step_metrics(run_id: str, metrics: dict, step: int):
    """Log training metrics for a single step to a specific run."""
    client = _get_client()
    for key, value in metrics.items():
        client.log_metric(run_id, key, value, step=step)


def log_eval_metrics(run_id: str, eval_results: dict):
    """Log final evaluation metrics to a trainer's run."""
    client = _get_client()
    for k, v in eval_results.items():
        if k == "hyperparameters":
            continue
        if isinstance(v, (int, float)):
            client.log_metric(run_id, f"eval_{k}", v)
        elif isinstance(v, t.Tensor):
            client.log_metric(run_id, f"eval_{k}", v.item())


def log_artifacts(run_id: str, save_dir: str):
    """Log ae.pt and config.json as artifacts for a trainer's run."""
    client = _get_client()
    ae_path = os.path.join(save_dir, "ae.pt")
    config_path = os.path.join(save_dir, "config.json")
    if os.path.exists(ae_path):
        client.log_artifact(run_id, ae_path)
    if os.path.exists(config_path):
        client.log_artifact(run_id, config_path)
