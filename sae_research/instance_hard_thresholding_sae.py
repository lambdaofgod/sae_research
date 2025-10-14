import torch
import torch.nn as nn

import sae_bench.custom_saes.base_sae as base_sae
from sae_research.modules import InstanceHardThresholding, SparseLeastSquares


class InstanceHardThresholdingSAE(base_sae.BaseSAE):
    """
    SAE using instance-level hard thresholding pursuit.

    Instead of selecting top-k features per token, this selects features
    per instance (across all tokens in a sequence), then solves a least
    squares problem to find the coefficients.
    """

    k: torch.Tensor

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)

        assert isinstance(k, int) and k > 0
        self.register_buffer("k", torch.tensor(k, dtype=torch.int, device=device))

        # Instantiate reusable modules
        self.hard_thresholding = InstanceHardThresholding()
        self.sparse_lstsq = SparseLeastSquares()

    def encode(self, x: torch.Tensor):
        """
        Encode activations using instance-level hard thresholding pursuit.

        Args:
            x: Input activations of shape (batch, seq_len, d_in)

        Returns:
            Encoded activations of shape (batch, seq_len, d_sae)
        """
        # Center the input by subtracting decoder bias
        x_centered = x - self.b_dec

        batch_size, seq_len, d_in = x_centered.shape

        # Initialize output
        encoded_acts = torch.zeros(
            batch_size, seq_len, self.W_dec.shape[0], device=x.device, dtype=x.dtype
        )

        # Process each instance (batch element) separately
        for i in range(batch_size):
            x_instance = x_centered[i]  # (seq_len, d_in)

            # Compute feature activations
            feature_acts = x_instance @ self.W_dec.T  # (seq_len, d_sae)

            # Find support across all tokens in this instance
            support = self.hard_thresholding(feature_acts, self.k.item())

            # Solve least squares on the support
            coeffs = self.sparse_lstsq(self.W_dec, x_instance, support)

            encoded_acts[i] = coeffs

        return encoded_acts

    def decode(self, feature_acts: torch.Tensor):
        """Standard linear decode."""
        return (feature_acts @ self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor):
        """Forward pass: encode then decode."""
        encoded = self.encode(x)
        reconstructed = self.decode(encoded)
        return reconstructed
