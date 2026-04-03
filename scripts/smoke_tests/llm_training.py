"""Quick test: 200 training steps using LLM activation buffer (no activault)."""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    from sae_research.training.cli_runner import run_sae_training
    from sae_research.training import config as demo_config

    # Use the non-instruct gemma config (no activault)
    model_name = "google/gemma-2-2b"
    sae_batch_size = demo_config.LLM_CONFIG[model_name].sae_batch_size
    steps = 320
    num_tokens = steps * sae_batch_size
    buffer_tokens = 32 * sae_batch_size  # match activault's 32-step refresh cadence

    run_sae_training(
        model_name=model_name,
        layer=13,
        save_dir="/tmp/llm_test",
        device="cuda",
        architectures=["top_k"],
        num_tokens=num_tokens,
        random_seeds=[0],
        dictionary_widths=[2**14],
        learning_rates=[1e-4],
        dry_run=False,
        use_wandb=False,
        save_checkpoints=False,
        buffer_tokens=buffer_tokens,
    )

    print("Done!")
