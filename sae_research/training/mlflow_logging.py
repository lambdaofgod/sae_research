"""
MLflow logging helpers for SAE training.

Provides a thin wrapper around mlflow to handle parent/child run hierarchy
and standardized param/metric logging for SAE sweeps.
"""

import os
from typing import Optional

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
) -> mlflow.ActiveRun:
    """Start a child MLflow run for one trainer within a sweep."""
    # Extract loggable params from the trainer config
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

    trainer_class_name = ""
    if "trainer" in trainer_config:
        trainer_cls = trainer_config["trainer"]
        if isinstance(trainer_cls, type):
            trainer_class_name = trainer_cls.__name__
        else:
            trainer_class_name = str(trainer_cls)

    dict_class_name = ""
    if "dict_class" in trainer_config:
        dict_cls = trainer_config["dict_class"]
        if isinstance(dict_cls, type):
            dict_class_name = dict_cls.__name__
        else:
            dict_class_name = str(dict_cls)

    run_name = f"trainer_{trainer_index}_{trainer_class_name}"

    run = mlflow.start_run(
        run_name=run_name,
        nested=True,
        parent_run_id=parent_run_id,
        tags={
            "trainer_index": str(trainer_index),
            "trainer_class": trainer_class_name,
            "dict_class": dict_class_name,
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

    mlflow.log_params(params)
    return run


def log_step_metrics(run_id: str, metrics: dict, step: int):
    """Log training metrics for a single step to a specific run."""
    with mlflow.start_run(run_id=run_id, nested=True):
        mlflow.log_metrics(metrics, step=step)


def log_eval_metrics(run_id: str, eval_results: dict):
    """Log final evaluation metrics to a trainer's run."""
    flat = {}
    for k, v in eval_results.items():
        if k == "hyperparameters":
            continue
        if isinstance(v, (int, float)):
            flat[f"eval_{k}"] = v
        elif isinstance(v, t.Tensor):
            flat[f"eval_{k}"] = v.item()
    if flat:
        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.log_metrics(flat)


def log_artifacts(run_id: str, save_dir: str):
    """Log ae.pt and config.json as artifacts for a trainer's run."""
    with mlflow.start_run(run_id=run_id, nested=True):
        ae_path = os.path.join(save_dir, "ae.pt")
        config_path = os.path.join(save_dir, "config.json")
        if os.path.exists(ae_path):
            mlflow.log_artifact(ae_path)
        if os.path.exists(config_path):
            mlflow.log_artifact(config_path)
