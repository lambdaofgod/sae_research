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

import itertools
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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


def main(config: str):
    """YAML-driven SAE training runner.

    Args:
        config: Path to YAML config file.
    """
    cfg = load_config(config)

    from datasets import config as ds_config

    ds_config.STREAMING_READ_MAX_RETRIES = 100  # pyrefly: ignore [bad-assignment]
    ds_config.STREAMING_READ_RETRY_INTERVAL = 20  # pyrefly: ignore [bad-assignment]

    from sae_research.training.cli_runner import run_sae_training, eval_saes
    from sae_research.training import config as training_config
    from sae_research.training import utils

    start_time = time.time()

    model_name = cfg["model_name"]
    architectures = cfg["architectures"]
    layers = cfg["layers"]

    sae_batch_size = training_config.LLM_CONFIG[model_name].sae_batch_size
    steps = int(cfg["num_tokens"] / sae_batch_size)

    all_mlflow_run_ids = []
    for architecture in architectures:
        trainer_path, dict_class_path = training_config.resolve_architecture(
            architecture
        )
        arch_sweep = training_config.get_architecture_sweep_params(architecture, steps)

        for layer in layers:
            save_dir = f"{cfg['save_dir']}_{model_name}_{architecture}".replace(
                "/", "_"
            )

            for seed, dict_width, lr, arch_params in itertools.product(
                cfg["random_seeds"],
                cfg["dictionary_widths"],
                cfg["learning_rates"],
                arch_sweep,
            ):
                run_id = run_sae_training(
                    model_name=model_name,
                    layer=layer,
                    save_dir=save_dir,
                    device=cfg["device"],
                    trainer=trainer_path,
                    dict_class=dict_class_path,
                    num_tokens=cfg["num_tokens"],
                    seed=seed,
                    dictionary_width=dict_width,
                    learning_rate=lr,
                    dry_run=cfg.get("dry_run", False),
                    mlflow_experiment=cfg.get("mlflow_experiment", ""),
                    save_checkpoints=cfg.get("save_checkpoints", False),
                    buffer_tokens=cfg.get("buffer_tokens", 250_000),
                    mixed_dataset=cfg.get("mixed_dataset", False),
                    remove_bos=cfg["remove_bos"],
                    max_activation_norm_multiple=cfg["max_activation_norm_multiple"],
                    **arch_params,
                )
                if run_id is not None:
                    all_mlflow_run_ids.append(run_id)

    ae_paths = utils.get_nested_folders(cfg["save_dir"])

    eval_saes(
        model_name,
        ae_paths,
        cfg["eval_num_inputs"],
        cfg["device"],
        overwrite_prev_results=True,
        mlflow_run_ids=all_mlflow_run_ids,
    )

    print(f"Total time: {time.time() - start_time}")

    hf_repo_id = cfg.get("hf_repo_id")
    if hf_repo_id:
        import huggingface_hub

        assert huggingface_hub.repo_exists(repo_id=hf_repo_id, repo_type="model")
        from sae_research.training.cli_runner import push_to_huggingface

        push_to_huggingface(cfg["save_dir"], hf_repo_id)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
