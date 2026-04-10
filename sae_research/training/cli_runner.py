import os

# I believe this environment variable should be set before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    architecture: str,
    num_tokens: int,
    random_seeds: list[int],
    dictionary_widths: list[int],
    learning_rates: list[float],
    dry_run: bool = False,
    use_mlflow: bool = True,
    mlflow_parent_run_id: str | None = None,
    save_checkpoints: bool = False,
    buffer_tokens: int = 250_000,
    mixed_dataset: bool = False,
    remove_bos: bool | None = None,
    max_activation_norm_multiple: float | None = None,
):
    if remove_bos is None:
        remove_bos = demo_config.remove_bos
    if max_activation_norm_multiple is None:
        max_activation_norm_multiple = demo_config.max_activation_norm_multiple

    random.seed(random_seeds[0])
    t.manual_seed(random_seeds[0])

    # model and data parameters
    context_length = demo_config.LLM_CONFIG[model_name].context_length

    llm_batch_size = demo_config.LLM_CONFIG[model_name].llm_batch_size
    sae_batch_size = demo_config.LLM_CONFIG[model_name].sae_batch_size
    dtype = demo_config.LLM_CONFIG[model_name].dtype

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

    # TODO: temporal should be declared by the architecture config, not hardcoded here
    _TEMPORAL_ARCHITECTURES = {
        demo_config.TrainerType.TEMPORAL_MATRYOSHKA_BATCH_TOP_K.value,
        demo_config.TrainerType.TEMPORAL_BATCH_TOP_K.value,
    }
    is_temporal = architecture in _TEMPORAL_ARCHITECTURES

    llm_config = demo_config.LLM_CONFIG[model_name]
    submodule_name = f"resid_post_layer_{layer}"
    io = "out"

    if is_temporal and llm_config.activault is not None:
        # Temporal Activault path: preserve sequence order for adjacent-token pairing
        from dictionary_learning.activault_s3_buffer import S3RCache, create_s3_client
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
                tokenizer,
                system_prompt_to_remove=qwen_system_prompt_to_remove,
                sequence_pack_pretrain=True,
                system_prompt_removal_freq=0.0,
                min_chars=context_length * 4,
            )
        else:
            generator = hf_sequence_packing_dataset_to_generator(
                tokenizer,
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

    trainer_configs = demo_config.get_trainer_configs(
        [architecture],
        learning_rates,
        random_seeds,
        activation_dim,
        dictionary_widths,
        model_name,
        device,
        layer,
        submodule_name,
        steps,
    )

    print(f"len trainer configs: {len(trainer_configs)}")
    assert len(trainer_configs) > 0

    save_dir = f"{save_dir}/{submodule_name}"

    if not dry_run:
        # Temporal SAEs don't support activation normalization (different data shape)
        normalize = not is_temporal
        mlflow_run_ids = []
        for config in trainer_configs:
            run_ids = trainSAE(
                data=activation_buffer,
                trainer_configs=[config],
                use_mlflow=use_mlflow,
                mlflow_parent_run_id=mlflow_parent_run_id,
                steps=steps,
                save_steps=save_steps,
                save_dir=save_dir,
                log_steps=log_steps,
                normalize_activations=normalize,
                verbose=False,
                autocast_dtype=t.bfloat16,
                backup_steps=1000,
            )
            mlflow_run_ids.extend(run_ids)
        return mlflow_run_ids
    return []


@t.no_grad()
def eval_saes(
    model_name: str,
    ae_paths: list[str],
    n_inputs: int,
    device: str,
    overwrite_prev_results: bool = False,
    transcoder: bool = False,
    mlflow_run_ids: list[str] | None = None,
) -> dict:
    random.seed(demo_config.random_seeds[0])
    t.manual_seed(demo_config.random_seeds[0])

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
            remove_bos=demo_config.remove_bos,
            max_activation_norm_multiple=demo_config.max_activation_norm_multiple,
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
    device: str = "cuda:0",
    mlflow: bool = True,
    dry_run: bool = False,
    save_checkpoints: bool = False,
    hf_repo_id: str | None = None,
    mixed_dataset: bool = False,
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
    if hf_repo_id:
        assert huggingface_hub.repo_exists(repo_id=hf_repo_id, repo_type="model")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    config.STREAMING_READ_MAX_RETRIES = 100  # pyrefly: ignore [bad-assignment]
    config.STREAMING_READ_RETRY_INTERVAL = 20  # pyrefly: ignore [bad-assignment]

    start_time = time.time()

    save_dir = f"{save_dir}_{model_name}_{architecture}".replace("/", "_")

    mlflow_parent_run_id = None
    mlflow_parent_run = None
    if mlflow:
        from sae_research.training.mlflow_logging import start_sweep_run

        mlflow_parent_run = start_sweep_run(
            experiment_name=mlflow_experiment,
            model_name=model_name,
            layers=[layer],
            architectures=[architecture],
            run_cfg={
                "num_tokens": demo_config.num_tokens,
                "save_dir": save_dir,
            },
        )
        mlflow_parent_run_id = mlflow_parent_run.info.run_id

    mlflow_run_ids = run_sae_training(
        model_name=model_name,
        layer=layer,
        save_dir=save_dir,
        device=device,
        architecture=architecture,
        num_tokens=demo_config.num_tokens,
        random_seeds=demo_config.random_seeds,
        dictionary_widths=demo_config.dictionary_widths,
        learning_rates=demo_config.learning_rates,
        dry_run=dry_run,
        use_mlflow=mlflow,
        mlflow_parent_run_id=mlflow_parent_run_id,
        save_checkpoints=save_checkpoints,
        mixed_dataset=mixed_dataset,
    )

    ae_paths = utils.get_nested_folders(save_dir)

    eval_saes(
        model_name,
        ae_paths,
        demo_config.eval_num_inputs,
        device,
        overwrite_prev_results=True,
        mlflow_run_ids=mlflow_run_ids,
    )

    if mlflow_parent_run is not None:
        import mlflow as mlflow_lib

        mlflow_lib.end_run()

    print(f"Total time: {time.time() - start_time}")

    if hf_repo_id:
        push_to_huggingface(save_dir, hf_repo_id)


if __name__ == "__main__":
    import fire

    fire.Fire(cli_main)
