"""
Training dictionaries
"""

import json
import os
from typing import Optional
from contextlib import nullcontext

import torch as t
from tqdm import tqdm

from sae_research.training.config import resolve_class
from sae_research.training.mlflow_logging import (
    start_trainer_run,
    end_trainer_run,
    log_step_metrics,
    log_artifacts,
)


def log_stats(
    trainers,
    step: int,
    act: t.Tensor,
    activations_split_by_head: bool,
    transcoder: bool,
    mlflow_run_ids: list[str] = [],
    verbose: bool = False,
):
    with t.no_grad():
        # quick hack to make sure all trainers get the same x
        z = act.clone()
        for i, trainer in enumerate(trainers):
            log = {}
            act = z.clone()
            if activations_split_by_head:  # x.shape: [batch, pos, n_heads, d_head]
                act = act[..., i, :]
            frac_variance_explained = None
            if not transcoder:
                act, act_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()
                # fraction of variance explained
                total_variance = t.var(act, dim=0).sum()
                residual_variance = t.var(act - act_hat, dim=0).sum()
                frac_variance_explained = 1 - residual_variance / total_variance
                log["frac_variance_explained"] = frac_variance_explained.item()
            else:  # transcoder
                x, x_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()

            if verbose:
                print(
                    f"Step {step}: L0 = {l0}, frac_variance_explained = {frac_variance_explained}"
                )

            # log parameters from training
            log.update(
                {
                    k: v.cpu().item() if isinstance(v, t.Tensor) else v
                    for k, v in losslog.items()
                }
            )
            log["l0"] = l0
            trainer_log = trainer.get_logging_parameters()
            for name, value in trainer_log.items():
                if isinstance(value, t.Tensor):
                    value = value.cpu().item()
                log[name] = value

            if mlflow_run_ids:
                log_step_metrics(mlflow_run_ids[i], log, step)


def get_norm_factor(data, steps: int) -> float:
    """Per Section 3.1, find a fixed scalar factor so activation vectors have unit mean squared norm.
    This is very helpful for hyperparameter transfer between different layers and models.
    Use more steps for more accurate results.
    https://arxiv.org/pdf/2408.05147

    If experiencing troubles with hyperparameter transfer between models, it may be worth instead normalizing to the square root of d_model.
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes"""
    total_mean_squared_norm: float | t.Tensor = 0
    count = 0

    for step, act_BD in enumerate(
        tqdm(data, total=steps, desc="Calculating norm factor")
    ):
        if step > steps:
            break

        count += 1
        mean_squared_norm = t.mean(t.sum(act_BD**2, dim=1))
        total_mean_squared_norm += mean_squared_norm

    average_mean_squared_norm = total_mean_squared_norm / count
    norm_factor = t.sqrt(average_mean_squared_norm).item()  # pyrefly: ignore [bad-argument-type]

    print(f"Average mean squared norm: {average_mean_squared_norm}")
    print(f"Norm factor: {norm_factor}")

    return norm_factor


def trainSAE(
    data,
    trainer_configs: list[dict],
    steps: int,
    mlflow_experiment: str = "",
    save_steps: Optional[list[int]] = None,
    save_dir: Optional[str] = None,
    log_steps: Optional[int] = None,
    activations_split_by_head: bool = False,
    transcoder: bool = False,
    run_cfg: dict = {},
    normalize_activations: bool = False,
    verbose: bool = False,
    device: str = "cuda",
    autocast_dtype: t.dtype = t.float32,
    backup_steps: Optional[int] = None,
):
    """
    Train SAEs using the given trainers

    If normalize_activations is True, the activations will be normalized to have unit mean squared norm.
    The autoencoders weights will be scaled before saving, so the activations don't need to be scaled during inference.
    This is very helpful for hyperparameter transfer between different layers and models.

    Setting autocast_dtype to t.bfloat16 provides a significant speedup with minimal change in performance.
    """

    device_type = "cuda" if "cuda" in device else "cpu"
    autocast_context = (
        nullcontext()
        if device_type == "cpu"
        else t.autocast(device_type=device_type, dtype=autocast_dtype)
    )

    trainers = []
    trainer_import_paths = []
    for i, config in enumerate(trainer_configs):
        config = config.copy()
        trainer_path = config.pop("trainer")
        trainer_class = resolve_class(trainer_path)
        config["dict_class"] = resolve_class(config["dict_class"])
        trainers.append(trainer_class(**config))
        trainer_import_paths.append(trainer_path)

    # Start MLflow runs per trainer
    mlflow_run_ids = []
    if mlflow_experiment:
        for i, trainer in enumerate(trainers):
            run = start_trainer_run(
                experiment_name=mlflow_experiment,
                trainer_config={
                    "trainer": trainer_import_paths[i],
                    **trainer.config,
                    **run_cfg,
                },
            )
            mlflow_run_ids.append(run.info.run_id)

    # make save dirs, export config
    if save_dir is not None:
        save_dirs = [
            os.path.join(save_dir, f"trainer_{i}") for i in range(len(trainer_configs))
        ]
        for trainer, dir in zip(trainers, save_dirs):
            os.makedirs(dir, exist_ok=True)
            tc = trainer.config
            config = {
                "format_version": 1,
                "model_name": tc.get("lm_name", ""),
                "hook_layer": tc.get("layer", ""),
                "dict_class": tc.get("dict_class", type(trainer.ae).__name__),
                "activation_dim": tc.get("activation_dim", 0),
                "dict_size": tc.get("dict_size", 0),
                "trainer": tc,
            }
            try:
                config["buffer"] = data.config
            except AttributeError:
                pass
            with open(os.path.join(dir, "config.json"), "w") as f:
                json.dump(config, f, indent=4)
    else:
        save_dirs = [None for _ in trainer_configs]

    norm_factor: float = 1.0
    if normalize_activations:
        norm_factor = get_norm_factor(data, steps=100)

        for trainer in trainers:
            trainer.config["norm_factor"] = norm_factor
            # Verify that all autoencoders have a scale_biases method
            trainer.ae.scale_biases(1.0)

    actual_steps = 0
    try:
        for step, act in enumerate(tqdm(data, total=steps)):
            act = act.to(dtype=autocast_dtype)

            if normalize_activations:
                act /= norm_factor

            if step >= steps:
                break

            actual_steps = step

            # logging
            if (
                (mlflow_experiment or verbose)
                and log_steps is not None
                and step % log_steps == 0
            ):
                log_stats(
                    trainers,
                    step,
                    act,
                    activations_split_by_head,
                    transcoder,
                    mlflow_run_ids=mlflow_run_ids,
                    verbose=verbose,
                )

            # saving
            if save_steps is not None and step in save_steps:
                for dir, trainer in zip(save_dirs, trainers):
                    if dir is None:
                        continue

                    if normalize_activations:
                        # Temporarily scale up biases for checkpoint saving
                        trainer.ae.scale_biases(norm_factor)

                    if not os.path.exists(os.path.join(dir, "checkpoints")):
                        os.mkdir(os.path.join(dir, "checkpoints"))

                    checkpoint = {
                        k: v.cpu() for k, v in trainer.ae.state_dict().items()
                    }
                    t.save(
                        checkpoint,
                        os.path.join(dir, "checkpoints", f"ae_{step}.pt"),
                    )

                    if normalize_activations:
                        trainer.ae.scale_biases(1 / norm_factor)

            # backup
            if backup_steps is not None and step % backup_steps == 0:
                for save_dir, trainer in zip(save_dirs, trainers):
                    if save_dir is None:
                        continue
                    # save the current state of the trainer for resume if training is interrupted
                    # this will be overwritten by the next checkpoint and at the end of training
                    backup = {
                        "step": step,
                        "ae": trainer.ae.state_dict(),
                        "optimizer": trainer.optimizer.state_dict(),
                        "config": trainer.config,
                    }
                    if normalize_activations:
                        backup["norm_factor"] = norm_factor
                    t.save(backup, os.path.join(save_dir, "ae.pt"))

            # training
            for trainer in trainers:
                with autocast_context:
                    trainer.update(step, act)
    except StopIteration:
        print(
            f"\nWARNING: Data exhausted at step {actual_steps} of {steps} requested. Saving current state."
        )

    # save final SAEs
    for save_dir, trainer in zip(save_dirs, trainers):
        if normalize_activations:
            trainer.ae.scale_biases(norm_factor)
        if save_dir is not None:
            checkpoint = {
                "step": actual_steps,
                "ae": {k: v.cpu() for k, v in trainer.ae.state_dict().items()},
                "config": trainer.config,
            }
            if normalize_activations:
                checkpoint["norm_factor"] = norm_factor
            t.save(checkpoint, os.path.join(save_dir, "ae.pt"))

    # Log final artifacts and mark runs as finished
    if mlflow_experiment:
        for run_id, dir in zip(mlflow_run_ids, save_dirs):
            if dir is not None:
                log_artifacts(run_id, dir)
            end_trainer_run(run_id)

    return mlflow_run_ids
