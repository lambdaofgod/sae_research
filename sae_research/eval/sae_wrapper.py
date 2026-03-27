"""
Generic wrapper for dictionary_learning SAEs to be compatible with sae_bench.

Wraps any autoencoder from the dictionary_learning library so it can be
passed to sae_bench evaluation functions as a (name, sae) tuple.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path

import sae_bench.custom_saes.base_sae as base_sae
from sae_research.training.utils import load_dictionary


# Map dictionary_learning trainer classes to sae_bench architecture names
_ARCHITECTURE_MAP = {
    "StandardTrainer": "standard",
    "PAnnealTrainer": "p_anneal",
    "TopKTrainer": "topk",
    "BatchTopKTrainer": "batch_topk",
    "MatryoshkaBatchTopKTrainer": "matryoshka_batch_topk",
    "GatedSAETrainer": "gated",
    "JumpReluTrainer": "jumprelu",
    "ThresholdingTopKTrainer": "thresholding_topk",
    "NestedThresholdingTopKTrainer": "nested_topk",
    "MatchingPursuitTrainer": "matching_pursuit",
    "NestedMatchingPursuitTrainer": "nested_matching_pursuit",
}


class DictionaryLearningSAEWrapper(base_sae.BaseSAE):
    """Wraps a dictionary_learning autoencoder for use with sae_bench.

    sae_bench expects SAEs with W_enc, W_dec, b_enc, b_dec parameters and
    encode/decode/forward methods. This wrapper delegates computation to
    the underlying autoencoder while exposing weights in sae_bench format.
    """

    def __init__(
        self,
        autoencoder: nn.Module,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        config_dict: dict,
    ):
        d_in = autoencoder.activation_dim
        d_sae = autoencoder.dict_size
        hook_name = f"blocks.{hook_layer}.hook_resid_post"

        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)

        self.autoencoder = autoencoder

        # Expose weights in sae_bench format.
        # dictionary_learning uses nn.Linear, so:
        #   encoder.weight shape: (dict_size, activation_dim)
        #   decoder.weight shape: (activation_dim, dict_size)
        # sae_bench expects:
        #   W_enc shape: (d_in, d_sae)
        #   W_dec shape: (d_sae, d_in)
        self.W_enc = nn.Parameter(autoencoder.encoder.weight.T.detach())
        self.W_dec = nn.Parameter(autoencoder.decoder.weight.T.detach())

        if hasattr(autoencoder.encoder, "bias") and autoencoder.encoder.bias is not None:
            self.b_enc = nn.Parameter(autoencoder.encoder.bias.detach())

        if hasattr(autoencoder, "b_dec"):
            self.b_dec = nn.Parameter(autoencoder.b_dec.detach())

        # Set architecture name from trainer class
        trainer_class = config_dict.get("trainer", {}).get("trainer_class", "")
        self.cfg.architecture = _ARCHITECTURE_MAP.get(trainer_class, trainer_class)

        training_tokens = config_dict.get("trainer", {}).get("training_tokens", 0)
        self.cfg.training_tokens = training_tokens

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(feature_acts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def load_sae_for_eval(
    sae_path: str,
    model_name: str,
    hook_layer: int | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[str, DictionaryLearningSAEWrapper]:
    """Load a dictionary_learning SAE and wrap it for sae_bench evaluation.

    Args:
        sae_path: Path to directory containing ae.pt and config.json
        model_name: Name of the language model (for sae_bench hook config)
        hook_layer: Layer number. Read from config if None.
        device: Device to load onto.
        dtype: Data type.

    Returns:
        (sae_name, wrapped_sae) tuple ready for sae_bench selected_saes list.
    """
    config_path = Path(sae_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    if hook_layer is None:
        hook_layer = config["trainer"]["layer"]

    autoencoder, _ = load_dictionary(sae_path, device)
    autoencoder = autoencoder.to(dtype=dtype)

    wrapped = DictionaryLearningSAEWrapper(
        autoencoder=autoencoder,
        model_name=model_name,
        hook_layer=hook_layer,
        device=torch.device(device),
        dtype=dtype,
        config_dict=config,
    )

    # Generate a name from the path
    path = Path(sae_path)
    sae_name = f"{path.parent.name}_{path.name}"

    return sae_name, wrapped
