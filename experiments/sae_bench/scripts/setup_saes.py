"""
Setup SAEs: Load baseline SAE and create custom variants (IHTP, MPSAE)
"""
import os
import sys
import yaml
import torch
import pickle
import fire
from pathlib import Path

import sae_bench.custom_saes.custom_sae_config as custom_sae_config
import sae_bench.custom_saes.relu_sae as relu_sae
import sae_bench.sae_bench_utils.general_utils as general_utils
from sae_bench.sae_bench_utils.sae_selection_utils import get_saes_from_regex
from sae_research.instance_sae import InstanceHardThresholdingPursuitSAE, MPSAE


def load_params():
    """Load parameters from params.yaml"""
    params_path = Path(__file__).parent.parent / "params.yaml"
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)


def setup_environment(params):
    """Setup device and environment"""
    device = general_utils.setup_environment()
    print(f"Using device: {device}")
    return device


def load_baseline_sae(params, device):
    """Load baseline SAE from HuggingFace"""
    print("\n=== Loading Baseline SAE ===")

    model_name = params['model']['name']
    torch_dtype_str = params['model']['torch_dtype']
    torch_dtype = getattr(torch, torch_dtype_str)

    repo_id = params['baseline_sae']['repo_id']
    filename = params['baseline_sae']['filename']
    hook_layer = params['baseline_sae']['hook_layer']

    print(f"Loading from {repo_id}/{filename}")

    sae = relu_sae.load_dictionary_learning_relu_sae(
        repo_id, filename, model_name, device, torch_dtype, layer=hook_layer
    )

    print(f"SAE dtype: {sae.dtype}, device: {sae.device}")

    d_sae, d_in = sae.W_dec.data.shape
    print(f"d_in: {d_in}, d_sae: {d_sae}")

    return sae, d_in, d_sae, hook_layer, torch_dtype, torch_dtype_str


def configure_baseline_sae(sae, params, model_name, d_in, d_sae, hook_name, hook_layer, str_dtype):
    """Configure baseline SAE with metadata"""
    print("\n=== Configuring Baseline SAE ===")

    sae.cfg = custom_sae_config.CustomSAEConfig(
        model_name, d_in=d_in, d_sae=d_sae,
        hook_name=hook_name, hook_layer=hook_layer
    )
    sae.cfg.dtype = str_dtype
    sae.cfg.architecture = "vanilla"
    sae.cfg.training_tokens = params['baseline_sae']['training_tokens']

    return sae


def create_ihtp_variants(sae, params, d_in, d_sae, model_name, hook_layer, device, torch_dtype, str_dtype):
    """Create IHTP SAE variants with different k values"""
    if not params['sae_variants']['ihtp']['enabled']:
        print("\nIHTP variants disabled, skipping...")
        return []

    print("\n=== Creating IHTP SAE Variants ===")
    ihtp_saes = []

    for k in params['sae_variants']['ihtp']['k_values']:
        print(f"Creating IHTP SAE with k={k}")

        ihtp_sae = InstanceHardThresholdingPursuitSAE(
            d_in=d_in, d_sae=d_sae, k=k,
            model_name=model_name,
            hook_layer=hook_layer,
            device=device, dtype=torch_dtype
        )

        # Copy weights from baseline
        ihtp_sae.W_dec.data = sae.W_dec.data.clone()
        ihtp_sae.b_dec.data = sae.b_dec.data.clone()
        ihtp_sae.W_enc.data = sae.W_enc.data.clone()
        ihtp_sae.b_enc.data = sae.b_enc.data.clone()

        # Configure
        ihtp_sae.cfg = custom_sae_config.CustomSAEConfig(
            model_name, d_in=d_in, d_sae=d_sae,
            hook_name=f"blocks.{hook_layer}.hook_resid_post",
            hook_layer=hook_layer
        )
        ihtp_sae.cfg.dtype = str_dtype
        ihtp_sae.cfg.architecture = "ihtp"
        ihtp_sae.cfg.training_tokens = params['baseline_sae']['training_tokens']

        ihtp_saes.append((f"ihtp_k{k}_layer{hook_layer}", ihtp_sae))

    return ihtp_saes


def create_mpsae_variants(sae, params, d_in, d_sae, model_name, hook_layer, device, torch_dtype, str_dtype):
    """Create MPSAE variants with different s values"""
    if not params['sae_variants']['mpsae']['enabled']:
        print("\nMPSAE variants disabled, skipping...")
        return []

    print("\n=== Creating MPSAE Variants ===")
    mpsaes = []

    for s in params['sae_variants']['mpsae']['s_values']:
        print(f"Creating MPSAE with s={s}")

        mpsae = MPSAE(
            d_in=d_in, d_sae=d_sae, s=s,
            model_name=model_name,
            hook_layer=hook_layer,
            device=device, dtype=torch_dtype
        )

        # Copy weights from baseline
        mpsae.W_dec.data = sae.W_dec.data.clone()
        mpsae.b_dec.data = sae.b_dec.data.clone()
        mpsae.W_enc.data = sae.W_enc.data.clone()
        mpsae.b_enc.data = sae.b_enc.data.clone()

        # Configure
        mpsae.cfg = custom_sae_config.CustomSAEConfig(
            model_name, d_in=d_in, d_sae=d_sae,
            hook_name=f"blocks.{hook_layer}.hook_resid_post",
            hook_layer=hook_layer
        )
        mpsae.cfg.dtype = str_dtype
        mpsae.cfg.architecture = "mpsae"
        mpsae.cfg.training_tokens = params['baseline_sae']['training_tokens']

        mpsaes.append((f"mpsae_s{s}_layer{hook_layer}", mpsae))

    return mpsaes


def get_comparison_saes(params):
    """Load baseline comparison SAEs from SAE Bench"""
    if not params['comparison_saes']['enabled']:
        print("\nComparison SAEs disabled, skipping...")
        return []

    print("\n=== Loading Comparison SAEs ===")

    sae_regex_pattern = params['comparison_saes']['sae_regex_pattern']
    sae_block_pattern = params['comparison_saes']['sae_block_pattern']

    baseline_saes = get_saes_from_regex(sae_regex_pattern, sae_block_pattern)
    print(f"Found {len(baseline_saes)} comparison SAE(s)")

    return baseline_saes


def save_saes(custom_saes, output_dir):
    """Save SAE objects and metadata"""
    print(f"\n=== Saving SAEs to {output_dir} ===")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save custom SAEs
    for sae_id, sae in custom_saes:
        sae_path = output_path / f"{sae_id}.pkl"
        with open(sae_path, 'wb') as f:
            pickle.dump(sae, f)
        print(f"Saved {sae_id}")

    # Save SAE list for downstream stages
    sae_list = [sae_id for sae_id, _ in custom_saes]
    list_path = output_path / "custom_sae_list.txt"
    with open(list_path, 'w') as f:
        f.write('\n'.join(sae_list))

    print(f"Saved {len(custom_saes)} custom SAE(s)")


def main(output_dir: str):
    """Setup SAEs and save to output directory

    Args:
        output_dir: Directory to save SAE weights and metadata
    """
    # Load parameters
    params = load_params()

    # Setup environment
    device = setup_environment(params)

    # Load baseline SAE
    sae, d_in, d_sae, hook_layer, torch_dtype, str_dtype = load_baseline_sae(params, device)

    # Configure baseline SAE
    model_name = params['model']['name']
    hook_name = f"blocks.{hook_layer}.hook_resid_post"
    baseline_filename = params['baseline_sae']['filename']
    unique_custom_sae_id = baseline_filename.replace("/", "_").replace(".", "_")

    sae = configure_baseline_sae(
        sae, params, model_name, d_in, d_sae, hook_name, hook_layer, str_dtype
    )

    # Create variant SAEs
    ihtp_saes = create_ihtp_variants(
        sae, params, d_in, d_sae, model_name, hook_layer, device, torch_dtype, str_dtype
    )

    mpsaes = create_mpsae_variants(
        sae, params, d_in, d_sae, model_name, hook_layer, device, torch_dtype, str_dtype
    )

    # Get comparison SAEs
    baseline_saes = get_comparison_saes(params)

    # Collect all custom SAEs
    custom_saes = [(unique_custom_sae_id, sae)] + ihtp_saes + mpsaes

    # Save SAEs
    save_saes(custom_saes, output_dir)

    # Save comparison SAE metadata
    if baseline_saes:
        comparison_path = Path(output_dir) / "comparison_saes.pkl"
        with open(comparison_path, 'wb') as f:
            pickle.dump(baseline_saes, f)
        print(f"\nSaved comparison SAE metadata")

    print("\n=== Setup Complete ===")
    print(f"Total custom SAEs: {len(custom_saes)}")
    print(f"Comparison SAEs: {len(baseline_saes)}")


if __name__ == "__main__":
    fire.Fire(main)
