import os

# I believe this environment variable should be set before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import itertools
import random
import json
import time
import huggingface_hub
from datasets import config
from transformers import AutoTokenizer

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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir", type=str, required=True, help="where to store sweep"
    )
    parser.add_argument("--mlflow", default=True, action=argparse.BooleanOptionalAction, help="log to MLflow (default: True)")
    parser.add_argument("--dry_run", action="store_true", help="dry run sweep")
    parser.add_argument(
        "--save_checkpoints", action="store_true", help="save checkpoints"
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", required=True, help="layers to train SAE on"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="which language model to use",
    )
    parser.add_argument(
        "--architectures",
        type=str,
        nargs="+",
        choices=[e.value for e in demo_config.TrainerType],
        required=True,
        help="which SAE architectures to train",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="device to train on"
    )
    parser.add_argument(
        "--hf_repo_id", type=str, help="Hugging Face repo ID to push results to"
    )
    parser.add_argument(
        "--mixed_dataset", action="store_true", help="use mixed dataset"
    )

    args = parser.parse_args()
    return args


def run_sae_training(
    model_name: str,
    layer: int,
    save_dir: str,
    device: str,
    architectures: list,
    num_tokens: int,
    random_seeds: list[int],
    dictionary_widths: list[int],
    learning_rates: list[float],
    dry_run: bool = False,
    use_mlflow: bool = True,
    mlflow_parent_run_id: str = None,
    save_checkpoints: bool = False,
    buffer_tokens: int = 250_000,
    mixed_dataset: bool = False,
):
    random.seed(demo_config.random_seeds[0])
    t.manual_seed(demo_config.random_seeds[0])

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

    _TEMPORAL_ARCHITECTURES = {
        demo_config.TrainerType.TEMPORAL_MATRYOSHKA_BATCH_TOP_K.value,
        demo_config.TrainerType.TEMPORAL_BATCH_TOP_K.value,
    }
    is_temporal = bool(set(architectures) & _TEMPORAL_ARCHITECTURES)

    llm_config = demo_config.LLM_CONFIG[model_name]
    submodule_name = f"resid_post_layer_{layer}"
    io = "out"

    if is_temporal and llm_config.activault is not None:
        # Temporal Activault path: preserve sequence order for adjacent-token pairing
        from dictionary_learning.activault_s3_buffer import S3RCache, create_s3_client
        import os
        s3_client = create_s3_client(
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("S3_ENDPOINT_URL"),
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
        activation_buffer = TemporalS3Buffer(cache, batch_size=sae_batch_size, device=device)
        activation_dim = cache.metadata["shape"][-1]
        print(f"Using temporal activault: prefix={llm_config.activault.s3_prefix}, "
              f"activation_dim={activation_dim}, dtype={cache.metadata['dtype']}")
    elif is_temporal and llm_config.activault is None:
        raise NotImplementedError(
            "Temporal SAE training from LLM forward passes is not yet supported. "
            "Use Activault pre-computed activations instead."
        )
    elif llm_config.activault is not None:
        # Activault path: stream pre-computed activations from S3
        activation_buffer, metadata = utils.create_activault_buffer(
            llm_config.activault, sae_batch_size=sae_batch_size, device=device,
        )
        activation_dim = metadata["shape"][-1]
        print(f"Using activault: prefix={llm_config.activault.s3_prefix}, "
              f"activation_dim={activation_dim}, dtype={metadata['dtype']}")
    else:
        # LLM path: run forward passes to generate activations
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=dtype
        )

        model = utils.truncate_model(model, layer)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        submodule = utils.get_submodule(model, layer)
        activation_dim = model.config.hidden_size

        if "Qwen" in model_name and demo_config.remove_bos:
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
            remove_bos=demo_config.remove_bos,
            max_activation_norm_multiple=demo_config.max_activation_norm_multiple,
        )

    trainer_configs = demo_config.get_trainer_configs(
        architectures,
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
        # actually run the sweep
        # Temporal SAEs don't support activation normalization (different data shape)
        normalize = not is_temporal
        mlflow_run_ids = trainSAE(
            data=activation_buffer,
            trainer_configs=trainer_configs,
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
    mlflow_run_ids: list[str] = None,
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
        eval_results["hyperparameters"] = hyperparameters

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


if __name__ == "__main__":
    """python runner.py --save_dir run2 --model_name EleutherAI/pythia-70m-deduped --layers 3 --architectures standard jump_relu batch_top_k top_k gated
    python runner.py --save_dir run3 --model_name google/gemma-2-2b --layers 12 --architectures standard top_k"""
    args = get_args()

    hf_repo_id = args.hf_repo_id

    if hf_repo_id:
        assert huggingface_hub.repo_exists(repo_id=hf_repo_id, repo_type="model")

    # This prevents random CUDA out of memory errors
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Rarely I have internet issues on cloud GPUs and then the streaming read fails
    # Hopefully the outage is shorter than 100 * 20 seconds
    config.STREAMING_READ_MAX_RETRIES = 100
    config.STREAMING_READ_RETRY_INTERVAL = 20

    start_time = time.time()

    save_dir = (
        f"{args.save_dir}_{args.model_name}_{'_'.join(args.architectures)}".replace(
            "/", "_"
        )
    )

    # Start MLflow parent run for the sweep
    mlflow_parent_run_id = None
    mlflow_parent_run = None
    if args.mlflow:
        from sae_research.training.mlflow_logging import start_sweep_run
        mlflow_parent_run = start_sweep_run(
            experiment_name=demo_config.mlflow_experiment,
            model_name=args.model_name,
            layers=args.layers,
            architectures=args.architectures,
            run_cfg={
                "num_tokens": demo_config.num_tokens,
                "save_dir": save_dir,
            },
        )
        mlflow_parent_run_id = mlflow_parent_run.info.run_id

    all_mlflow_run_ids = []
    for layer in args.layers:
        mlflow_run_ids = run_sae_training(
            model_name=args.model_name,
            layer=layer,
            save_dir=save_dir,
            device=args.device,
            architectures=args.architectures,
            num_tokens=demo_config.num_tokens,
            random_seeds=demo_config.random_seeds,
            dictionary_widths=demo_config.dictionary_widths,
            learning_rates=demo_config.learning_rates,
            dry_run=args.dry_run,
            use_mlflow=args.mlflow,
            mlflow_parent_run_id=mlflow_parent_run_id,
            save_checkpoints=args.save_checkpoints,
            mixed_dataset=args.mixed_dataset,
        )
        all_mlflow_run_ids.extend(mlflow_run_ids)

    ae_paths = utils.get_nested_folders(save_dir)

    eval_saes(
        args.model_name,
        ae_paths,
        demo_config.eval_num_inputs,
        args.device,
        overwrite_prev_results=True,
        mlflow_run_ids=all_mlflow_run_ids,
    )

    # End the parent sweep run
    if mlflow_parent_run is not None:
        import mlflow
        mlflow.end_run()

    print(f"Total time: {time.time() - start_time}")

    if hf_repo_id:
        push_to_huggingface(save_dir, hf_repo_id)
