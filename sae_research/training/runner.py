"""YAML-driven training runner.

Usage:
    python -m sae_research.training.runner --config run.yaml

The YAML config is the complete, self-contained spec for a training run.
No hidden defaults, no demo_config globals. What you pass is what runs.

Example config:

    model_name: google/gemma-2-2b-it
    layers: [13]
    architectures: [batch_top_k]
    save_dir: sae_run
    device: cuda:0

    num_tokens: 200_000_000
    random_seeds: [0]
    dictionary_widths: [16384]
    learning_rates: [0.0001]

    mlflow: true
    mlflow_experiment: sae_training
    save_checkpoints: false
    mixed_dataset: false

    eval_num_inputs: 200
    remove_bos: true
    max_activation_norm_multiple: 10
    buffer_tokens: 250000
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import time

import yaml


_REQUIRED_KEYS = [
    "model_name",
    "layers",
    "architectures",
    "save_dir",
    "device",
    "num_tokens",
    "random_seeds",
    "dictionary_widths",
    "learning_rates",
    "mlflow_experiment",
    "eval_num_inputs",
    "remove_bos",
    "max_activation_norm_multiple",
]


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    return cfg


def main():
    parser = argparse.ArgumentParser(description="YAML-driven SAE training runner")
    parser.add_argument(
        "--config", type=str, required=True, help="path to YAML config file"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    from datasets import config as ds_config

    ds_config.STREAMING_READ_MAX_RETRIES = 100  # pyrefly: ignore [bad-assignment]
    ds_config.STREAMING_READ_RETRY_INTERVAL = 20  # pyrefly: ignore [bad-assignment]

    from sae_research.training.cli_runner import run_sae_training, eval_saes
    from sae_research.training import utils

    start_time = time.time()

    model_name = cfg["model_name"]
    architectures = cfg["architectures"]
    layers = cfg["layers"]
    save_dir = f"{cfg['save_dir']}_{model_name}_{'_'.join(architectures)}".replace(
        "/", "_"
    )

    use_mlflow = cfg.get("mlflow", True)
    mlflow_parent_run_id = None
    mlflow_parent_run = None
    if use_mlflow:
        from sae_research.training.mlflow_logging import start_sweep_run

        mlflow_parent_run = start_sweep_run(
            experiment_name=cfg["mlflow_experiment"],
            model_name=model_name,
            layers=layers,
            architectures=architectures,
            run_cfg={
                "num_tokens": cfg["num_tokens"],
                "save_dir": save_dir,
            },
        )
        mlflow_parent_run_id = mlflow_parent_run.info.run_id

    all_mlflow_run_ids = []
    for layer in layers:
        mlflow_run_ids = run_sae_training(
            model_name=model_name,
            layer=layer,
            save_dir=save_dir,
            device=cfg["device"],
            architectures=architectures,
            num_tokens=cfg["num_tokens"],
            random_seeds=cfg["random_seeds"],
            dictionary_widths=cfg["dictionary_widths"],
            learning_rates=cfg["learning_rates"],
            dry_run=cfg.get("dry_run", False),
            use_mlflow=use_mlflow,
            mlflow_parent_run_id=mlflow_parent_run_id,
            save_checkpoints=cfg.get("save_checkpoints", False),
            buffer_tokens=cfg.get("buffer_tokens", 250_000),
            mixed_dataset=cfg.get("mixed_dataset", False),
            remove_bos=cfg["remove_bos"],
            max_activation_norm_multiple=cfg["max_activation_norm_multiple"],
        )
        all_mlflow_run_ids.extend(mlflow_run_ids)

    ae_paths = utils.get_nested_folders(save_dir)

    eval_saes(
        model_name,
        ae_paths,
        cfg["eval_num_inputs"],
        cfg["device"],
        overwrite_prev_results=True,
        mlflow_run_ids=all_mlflow_run_ids,
    )

    if mlflow_parent_run is not None:
        import mlflow

        mlflow.end_run()

    print(f"Total time: {time.time() - start_time}")

    hf_repo_id = cfg.get("hf_repo_id")
    if hf_repo_id:
        import huggingface_hub

        assert huggingface_hub.repo_exists(repo_id=hf_repo_id, repo_type="model")
        from sae_research.training.cli_runner import push_to_huggingface

        push_to_huggingface(save_dir, hf_repo_id)


if __name__ == "__main__":
    main()
