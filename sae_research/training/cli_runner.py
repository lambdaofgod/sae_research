import os

# I believe this environment variable should be set before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import itertools

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import json
import time
import huggingface_hub
from datasets import config

from sae_research.training import config as demo_config

from dictionary_learning.utils import (
    hf_dataset_to_generator,
    hf_mixed_dataset_to_generator,
    hf_sequence_packing_dataset_to_generator,
)
from dictionary_learning.pytorch_buffer import ActivationBuffer
from dictionary_learning.evaluation import evaluate
from sae_research.training.train import trainSAE
from sae_research.training import utils
from sae_research.training.temporal_buffer import TemporalS3Buffer


def run_sae_training(
    model_name: str,
    layer: int,
    save_dir: str,
    device: str,
    trainer: str,
    dict_class: str,
    num_tokens: int,
    seed: int,
    dictionary_width: int,
    learning_rate: float,
    dry_run: bool = False,
    mlflow_experiment: str = "",
    save_checkpoints: bool = False,
    buffer_tokens: int = 250_000,
    mixed_dataset: bool = False,
    remove_bos: bool = True,
    max_activation_norm_multiple: float = 10,
    **arch_params,
) -> str | None:
    """Train a single SAE configuration.

    One call = one config = one buffer = one trainSAE() call.
    trainer and dict_class are dotted import path strings.
    Architecture-specific params (k, l1_penalty, temporal, etc.) go in **arch_params.
    """
    random.seed(seed)
    t.manual_seed(seed)

    # model and data parameters
    llm_config = demo_config.LLM_CONFIG[model_name]
    context_length = llm_config.context_length
    llm_batch_size = llm_config.llm_batch_size
    sae_batch_size = llm_config.sae_batch_size
    dtype = llm_config.dtype

    num_buffer_inputs = buffer_tokens // context_length
    print(f"buffer_size: {num_buffer_inputs}, buffer_size_in_tokens: {buffer_tokens}")

    log_steps = 100  # Log metrics every log_steps

    steps = int(num_tokens / sae_batch_size)  # Total number of batches to train

    if save_checkpoints:
        # Creates checkpoints at 0.0%, 0.1%, 0.316%, 1%, 3.16%, 10%, 31.6%, 100% of training
        desired_checkpoints = t.logspace(-3, 0, 7).tolist()
        desired_checkpoints = [0.0] + desired_checkpoints[:-1]
        desired_checkpoints.sort()
        print(f"desired_checkpoints: {desired_checkpoints}")

        save_steps = [int(steps * step) for step in desired_checkpoints]
        save_steps.sort()
        print(f"save_steps: {save_steps}")
    else:
        save_steps = None

    # Temporal architectures have "temporal" as a required config field,
    # so its presence in arch_params signals the buffer type.
    is_temporal = "temporal" in arch_params

    submodule_name = f"resid_post_layer_{layer}"
    io = "out"

    if is_temporal and llm_config.activault is not None:
        # Temporal Activault path: preserve sequence order for adjacent-token pairing
        from dictionary_learning.activault_s3_buffer import create_s3_client
        from sae_research.training.resilient_s3_cache import (
            ResilientS3RCache as S3RCache,
        )
        import os

        s3_client = create_s3_client(
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL")
            or os.environ.get("S3_ENDPOINT_URL"),
        )
        cache = S3RCache(
            s3_client=s3_client,
            s3_prefix=llm_config.activault.s3_prefix,
            bucket_name=llm_config.activault.s3_bucket,
            buffer_size=llm_config.activault.s3_buffer_size,
            n_workers=llm_config.activault.s3_workers,
            concurrency=llm_config.activault.s3_concurrency,
            device=device,
            return_ids=True,
            shuffle=False,
        )
        activation_buffer = TemporalS3Buffer(
            cache, batch_size=sae_batch_size, device=device
        )
        activation_dim = cache.metadata["shape"][-1]
        print(
            f"Using temporal activault: prefix={llm_config.activault.s3_prefix}, "
            f"activation_dim={activation_dim}, dtype={cache.metadata['dtype']}"
        )
    elif is_temporal and llm_config.activault is None:
        raise NotImplementedError(
            "Temporal SAE training from LLM forward passes is not yet supported. "
            "Use Activault pre-computed activations instead."
        )
    elif llm_config.activault is not None:
        # Activault path: stream pre-computed activations from S3
        activation_buffer, metadata = utils.create_activault_buffer(
            llm_config.activault,
            sae_batch_size=sae_batch_size,
            device=device,
        )
        activation_dim = metadata["shape"][-1]
        print(
            f"Using activault: prefix={llm_config.activault.s3_prefix}, "
            f"activation_dim={activation_dim}, dtype={metadata['dtype']}"
        )
    else:
        # LLM path: run forward passes to generate activations
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=dtype
        )

        model = utils.truncate_model(model, layer)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        submodule = utils.get_submodule(model, layer)
        activation_dim = model.config.hidden_size

        if "Qwen" in model_name and remove_bos:
            print(
                "\n\nWARNING: Qwen models do not have a bos token, we will remove the first non-pad token"
            )

        if mixed_dataset:
            qwen_system_prompt_to_remove = None

            generator = hf_mixed_dataset_to_generator(
                tokenizer,  # pyrefly: ignore [bad-argument-type]
                system_prompt_to_remove=qwen_system_prompt_to_remove,
                sequence_pack_pretrain=True,
                system_prompt_removal_freq=0.0,
                min_chars=context_length * 4,
            )
        else:
            generator = hf_sequence_packing_dataset_to_generator(
                tokenizer,  # pyrefly: ignore [bad-argument-type]
                min_chars=context_length * 4,
            )

        activation_buffer = ActivationBuffer(
            generator,
            model,
            submodule,
            n_ctxs=num_buffer_inputs,
            ctx_len=context_length,
            refresh_batch_size=llm_batch_size,
            out_batch_size=sae_batch_size,
            io=io,
            d_submodule=activation_dim,
            device=device,
            add_special_tokens=False,
            remove_bos=remove_bos,
            max_activation_norm_multiple=max_activation_norm_multiple,  # pyrefly: ignore [bad-argument-type]
        )

    trainer_config = demo_config.build_trainer_config(
        trainer=trainer,
        dict_class=dict_class,
        activation_dim=activation_dim,
        dict_size=dictionary_width,
        seed=seed,
        lr=learning_rate,
        steps=steps,
        device=device,
        layer=layer,
        model_name=model_name,
        submodule_name=submodule_name,
        **arch_params,
    )

    save_dir = f"{save_dir}/{submodule_name}"

    if dry_run:
        return None

    # Temporal SAEs don't support activation normalization (different data shape)
    normalize = not is_temporal
    run_ids = trainSAE(
        data=activation_buffer,
        trainer_configs=[trainer_config],
        mlflow_experiment=mlflow_experiment,
        steps=steps,
        save_steps=save_steps,
        save_dir=save_dir,
        log_steps=log_steps,
        normalize_activations=normalize,
        verbose=False,
        autocast_dtype=t.bfloat16,
        backup_steps=1000,
    )
    return run_ids[0] if run_ids else None


@t.no_grad()
def eval_saes(
    model_name: str,
    ae_paths: list[str],
    n_inputs: int,
    device: str,
    overwrite_prev_results: bool = False,
    transcoder: bool = False,
    mlflow_run_ids: list[str] | None = None,
    random_seed: int = 0,
    remove_bos: bool = True,
    max_activation_norm_multiple: float = 10,
) -> dict:
    random.seed(random_seed)
    t.manual_seed(random_seed)

    if transcoder:
        io = "in_and_out"
    else:
        io = "out"

    context_length = demo_config.LLM_CONFIG[model_name].context_length
    llm_batch_size = demo_config.LLM_CONFIG[model_name].llm_batch_size
    loss_recovered_batch_size = max(llm_batch_size // 5, 1)
    sae_batch_size = loss_recovered_batch_size * context_length
    dtype = demo_config.LLM_CONFIG[model_name].dtype

    max_layer = 0

    for ae_path in ae_paths:
        config_path = f"{ae_path}/config.json"

        with open(config_path, "r") as f:
            config = json.load(f)

        layer = config["trainer"]["layer"]
        max_layer = max(max_layer, layer)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=dtype
    )

    model = utils.truncate_model(model, max_layer)

    buffer_size = n_inputs
    io = "out"
    n_batches = n_inputs // loss_recovered_batch_size

    generator = hf_dataset_to_generator("monology/pile-uncopyrighted")

    input_strings = []
    for i, example in enumerate(generator):
        input_strings.append(example)
        if i > n_inputs * 5 * llm_batch_size:
            break

    eval_results = {}

    for idx, ae_path in enumerate(ae_paths):
        output_filename = f"{ae_path}/eval_results.json"
        if not overwrite_prev_results:
            if os.path.exists(output_filename):
                print(f"Skipping {ae_path} as eval results already exist")
                continue

        dictionary, config = utils.load_dictionary(ae_path, device)
        dictionary = dictionary.to(dtype=model.dtype)

        layer = config["trainer"]["layer"]
        submodule = utils.get_submodule(model, layer)

        activation_dim = config["trainer"]["activation_dim"]

        activation_buffer = ActivationBuffer(
            iter(input_strings),
            model,
            submodule,
            n_ctxs=buffer_size,
            ctx_len=context_length,
            refresh_batch_size=llm_batch_size,
            out_batch_size=sae_batch_size,
            io=io,
            d_submodule=activation_dim,
            device=device,
            remove_bos=remove_bos,
            max_activation_norm_multiple=max_activation_norm_multiple,  # pyrefly: ignore [bad-argument-type]
        )

        eval_results = evaluate(
            dictionary,
            activation_buffer,
            context_length,
            loss_recovered_batch_size,
            io=io,
            device=device,
            n_batches=n_batches,
        )

        hyperparameters = {
            "n_inputs": n_inputs,
            "context_length": context_length,
        }
        eval_results["hyperparameters"] = hyperparameters  # pyrefly: ignore [unsupported-operation]

        print(eval_results)

        with open(output_filename, "w") as f:
            json.dump(eval_results, f)

        # Log eval metrics to the corresponding MLflow child run
        if mlflow_run_ids and idx < len(mlflow_run_ids):
            from sae_research.training.mlflow_logging import log_eval_metrics

            log_eval_metrics(mlflow_run_ids[idx], eval_results)

    # return the final eval_results for testing purposes
    return eval_results


def push_to_huggingface(save_dir: str, repo_id: str):
    api = huggingface_hub.HfApi()

    api.upload_folder(
        folder_path=save_dir,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=save_dir,
    )


def cli_main(
    save_dir: str,
    model_name: str,
    layer: int,
    architecture: str,
    mlflow_experiment: str,
    num_tokens: int = 200_000_000,
    random_seeds: list[int] = [0],
    dictionary_widths: list[int] = [16384],
    learning_rates: list[float] = [0.0001],
    device: str = "cuda:0",
    dry_run: bool = False,
    save_checkpoints: bool = False,
    hf_repo_id: str | None = None,
    mixed_dataset: bool = False,
    remove_bos: bool = True,
    max_activation_norm_multiple: float = 10,
):
    """Train SAEs for one model/layer/architecture.

    Prefer the YAML-driven runner (python -m sae_research.training.runner) for
    new workflows. This CLI preserves the legacy interface.

    Usage:
        python -m sae_research.training.cli_runner \\
            --save_dir=run2 \\
            --model_name=EleutherAI/pythia-70m-deduped \\
            --layer=3 \\
            --architecture=batch_top_k \\
            --mlflow_experiment=my_sweep
    """
    from sae_research.training.mlflow_logging import configure_tracking_uri

    configure_tracking_uri()

    if hf_repo_id:
        assert huggingface_hub.repo_exists(repo_id=hf_repo_id, repo_type="model")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    config.STREAMING_READ_MAX_RETRIES = 100  # pyrefly: ignore [bad-assignment]
    config.STREAMING_READ_RETRY_INTERVAL = 20  # pyrefly: ignore [bad-assignment]

    start_time = time.time()

    save_dir = f"{save_dir}_{model_name}_{architecture}".replace("/", "_")

    trainer_path, dict_class_path = demo_config.resolve_architecture(architecture)
    sae_batch_size = demo_config.LLM_CONFIG[model_name].sae_batch_size
    steps = int(num_tokens / sae_batch_size)
    arch_sweep = demo_config.get_architecture_sweep_params(architecture, steps)

    mlflow_run_ids = []
    for seed, dict_width, lr, arch_params in itertools.product(
        random_seeds,
        dictionary_widths,
        learning_rates,
        arch_sweep,
    ):
        run_id = run_sae_training(
            model_name=model_name,
            layer=layer,
            save_dir=save_dir,
            device=device,
            trainer=trainer_path,
            dict_class=dict_class_path,
            num_tokens=num_tokens,
            seed=seed,
            dictionary_width=dict_width,
            learning_rate=lr,
            dry_run=dry_run,
            mlflow_experiment=mlflow_experiment,
            save_checkpoints=save_checkpoints,
            mixed_dataset=mixed_dataset,
            remove_bos=remove_bos,
            max_activation_norm_multiple=max_activation_norm_multiple,
            **arch_params,
        )
        if run_id is not None:
            mlflow_run_ids.append(run_id)

    print(f"Total time: {time.time() - start_time}")

    if hf_repo_id:
        push_to_huggingface(save_dir, hf_repo_id)


if __name__ == "__main__":
    import fire

    fire.Fire(cli_main)
