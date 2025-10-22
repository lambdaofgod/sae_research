"""
Adapters for dictionary-learning SAEs to work with sae_bench.

This module provides wrappers and loaders for SAEs trained using the
dictionary-learning library, making them compatible with the sae_bench
evaluation framework.
"""

import json
import torch
import torch.nn as nn
from typing import Union, Optional, Dict, Any
from pathlib import Path

import sae_bench.custom_saes.base_sae as base_sae
import sae_bench.custom_saes.custom_sae_config as custom_sae_config

# Import the NestedThresholdingAutoEncoderTopK from the local file
from sae_research.thresholding_sae import NestedThresholdingAutoEncoderTopK


class NestedThresholdingTopKSAE(base_sae.BaseSAE):
    """
    Wrapper for NestedThresholdingAutoEncoderTopK to be compatible with sae_bench.

    This wrapper makes the nested thresholding SAE compatible with the sae_bench
    evaluation framework by providing the expected interface and attributes.
    """

    def __init__(
        self,
        nested_sae: NestedThresholdingAutoEncoderTopK,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str,
        config_dict: Dict[str, Any],
    ):
        # Initialize BaseSAE with dimensions
        # nested_sae.decoder is nn.Linear(dict_size, activation_dim)
        # So decoder.weight.shape is (activation_dim, dict_size)
        d_in, d_sae = nested_sae.decoder.weight.shape
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)

        # Store the wrapped SAE
        self.nested_sae = nested_sae
        self.config_dict = config_dict

        # Expose weights in the format sae_bench expects
        # Note: decoder.weight in nn.Linear is (out_features, in_features) = (d_in, d_sae)
        # But W_dec should be (d_sae, d_in)
        # Create as parameters so they're recognized by nn.Module
        self.W_dec = nn.Parameter(nested_sae.decoder.weight.T.detach())  # Transpose to get (d_sae, d_in)
        self.b_dec = nn.Parameter(nested_sae.b_dec.detach())

        # For compatibility, create W_enc and b_enc (even though thresholding uses decoder.T)
        # This matches how the thresholding SAE encodes: (x - b_dec) @ decoder.weight
        self.W_enc = nn.Parameter(nested_sae.decoder.weight.detach())  # (d_in, d_sae)
        self.b_enc = nn.Parameter(torch.zeros(d_sae, device=device, dtype=dtype))

        # Store k values for reference
        self.k_values = nested_sae.k_values
        self.max_k = nested_sae.max_k

        # Configure metadata
        self.cfg = custom_sae_config.CustomSAEConfig(
            model_name, d_in=d_in, d_sae=d_sae,
            hook_name=hook_name, hook_layer=hook_layer
        )
        self.cfg.dtype = str(dtype).replace('torch.', '')
        self.cfg.architecture = "nested_topk"

        # Add training metadata from config if available
        trainer_config = config_dict.get('trainer', {})
        self.cfg.training_tokens = trainer_config.get('training_tokens', 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode using the maximum k value (default behavior).

        Args:
            x: Input tensor of shape (batch, seq, d_in) or (n_tokens, d_in)

        Returns:
            Encoded activations of shape matching input batch dimensions + (d_sae,)
        """
        # The nested SAE's default encode() uses max_k
        return self.nested_sae.encode(x)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Standard decode."""
        return self.nested_sae.decode(feature_acts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode."""
        encoded = self.encode(x)
        reconstructed = self.decode(encoded)
        return reconstructed

    def create_single_k_variant(self, k: int):
        """
        Create a variant of this SAE that uses only a single k value.

        This is useful for evaluating each k value separately in sae_bench.

        Args:
            k: The k value to use

        Returns:
            A new SAE instance that uses only the specified k
        """
        if k not in self.k_values:
            raise ValueError(f"k={k} not in available k_values: {self.k_values}")

        # Create a single-k version by modifying the nested SAE's k_values
        single_k_sae = NestedThresholdingAutoEncoderTopK(
            activation_dim=self.nested_sae.activation_dim,
            dict_size=self.nested_sae.dict_size,
            k_values=[k]  # Only use this single k value
        )

        # Copy weights
        single_k_sae.decoder.weight.data = self.nested_sae.decoder.weight.data.clone()
        single_k_sae.b_dec.data = self.nested_sae.b_dec.data.clone()

        # Move to same device and dtype
        single_k_sae = single_k_sae.to(device=self.device, dtype=self.dtype)

        # Wrap in a new instance
        wrapper = NestedThresholdingTopKSAE(
            nested_sae=single_k_sae,
            model_name=self.cfg.model_name,
            hook_layer=self.cfg.hook_layer,
            device=self.device,
            dtype=self.dtype,
            hook_name=self.cfg.hook_name,
            config_dict=self.config_dict
        )

        # Update architecture to indicate this is a single-k variant
        wrapper.cfg.architecture = f"nested_topk_k{k}"

        return wrapper


def load_nested_thresholding_sae(
    path: str,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer: Optional[int] = None,
) -> NestedThresholdingTopKSAE:
    """
    Load a NestedThresholdingAutoEncoderTopK SAE from a local path.

    Args:
        path: Path to the SAE directory containing ae.pt and config.json
        model_name: Name of the language model (e.g., "pythia-70m-deduped")
        device: Device to load the SAE on
        dtype: Data type for the SAE
        layer: Layer number (optional, will be read from config if not provided)

    Returns:
        A wrapped NestedThresholdingTopKSAE compatible with sae_bench
    """
    path = Path(path)

    # Load config
    config_path = path / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extract layer from config if not provided
    trainer_config = config['trainer']
    if layer is None:
        layer = trainer_config['layer']
    else:
        # Validate layer matches config
        assert layer == trainer_config['layer'], \
            f"Provided layer {layer} doesn't match config layer {trainer_config['layer']}"

    # Load the SAE using from_pretrained
    ae_path = path / "ae.pt"
    k_values = trainer_config.get('k_values', [trainer_config.get('k')])

    # Load the nested SAE
    nested_sae = NestedThresholdingAutoEncoderTopK.from_pretrained(
        str(ae_path),
        k_values=k_values,
        device=str(device)
    )

    # Move to correct dtype
    nested_sae = nested_sae.to(dtype=dtype)

    # Create hook name
    submodule_name = trainer_config.get('submodule_name', f'resid_post_layer_{layer}')
    hook_name = f"blocks.{layer}.hook_resid_post"

    # Wrap in sae_bench-compatible class
    wrapped_sae = NestedThresholdingTopKSAE(
        nested_sae=nested_sae,
        model_name=model_name,
        hook_layer=layer,
        device=device,
        dtype=dtype,
        hook_name=hook_name,
        config_dict=config
    )

    return wrapped_sae