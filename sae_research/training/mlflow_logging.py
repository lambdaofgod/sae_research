"""
MLflow logging helpers for SAE training.

Provides a thin wrapper around mlflow to handle standardized param/metric
logging for SAE runs. Each training job creates one flat MLflow run;
the experiment name groups related runs.
"""

import os

import mlflow
import torch as t


def _build_run_name(trainer_class_name: str, trainer_config: dict) -> str:
    """Build a descriptive run name from trainer class and key hyperparams.

    E.g. BatchTopKTrainer_k64_d16384_lr0.0001_s0
    """
    parts = [trainer_class_name]

    # Architecture-specific sparsity param (pick the one that's present)
    for key in ("k", "target_l0", "l1_penalty", "initial_sparsity_penalty", "s"):
        if key in trainer_config:
            val = trainer_config[key]
            if isinstance(val, float) and val == int(val):
                val = int(val)
            parts.append(f"{key}{val}")
            break

    if "dict_size" in trainer_config:
        parts.append(f"d{trainer_config['dict_size']}")
    if "lr" in trainer_config:
        parts.append(f"lr{trainer_config['lr']}")
    if "seed" in trainer_config:
        parts.append(f"s{trainer_config['seed']}")

    return "_".join(parts)


def start_trainer_run(
    experiment_name: str,
    trainer_config: dict,
) -> mlflow.entities.Run:
    """Start a flat MLflow run for one trainer.

    Creates a run in the given experiment. All config keys are logged as
    params (no hardcoded allowlist).

    Returns an mlflow.entities.Run (not ActiveRun) — the run stays in
    RUNNING status until end_trainer_run() is called.
    """
    client = _get_client()

    trainer_path = str(trainer_config.get("trainer", ""))
    trainer_class_name = trainer_path.rsplit(".", 1)[-1] if trainer_path else ""

    dict_class_path = str(trainer_config.get("dict_class", ""))
    dict_class_name = dict_class_path.rsplit(".", 1)[-1] if dict_class_path else ""

    run_name = _build_run_name(trainer_class_name, trainer_config)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    run = client.create_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags={
            "trainer_class": trainer_class_name,
            "dict_class": dict_class_name,
        },
    )

    # Log all config keys as params (skip non-serializable values)
    params = {}
    for key, val in trainer_config.items():
        if isinstance(val, t.Tensor):
            val = val.item()
        elif isinstance(val, type):
            val = f"{val.__module__}.{val.__qualname__}"
        elif isinstance(val, list):
            val = str(val)
        elif isinstance(val, (int, float, str, bool)):
            pass
        else:
            continue
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
