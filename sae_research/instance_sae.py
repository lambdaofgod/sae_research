import torch
import torch.nn as nn

import sae_bench.custom_saes.base_sae as base_sae
from sae_research.modules import (
    HardThresholdingSupportSelector,
    HardThresholding,
    SparseLeastSquares,
)
from abc import ABC, abstractmethod


class InstanceSparseCodingSAE:
    """
    SAE using instance-level hard thresholding pursuit.

    Instead of selecting top-k features per token, this selects features
    per instance (across all tokens in a sequence), then solves a least
    squares problem to find the coefficients.
    """

    instance_support_selector: nn.Module
    instance_lstsq: nn.Module
    thresholding: nn.Module
    W_dec: torch.Tensor
    b_dec: torch.Tensor
    b_enc: torch.Tensor

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
            support = self.instance_support_selector(feature_acts, self.k.item())

            # Solve least squares on the support
            coeffs = self.instance_lstsq(self.W_dec, x_instance, support)
            coeffs = self.thresholding(coeffs)

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


class InstanceHardThresholdingPursuitSAE(InstanceSparseCodingSAE, base_sae.BaseSAE):

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
        self.instance_support_selector = HardThresholdingSupportSelector()
        self.instance_lstsq = SparseLeastSquares()
        self.thresholding = nn.Identity()


class MPSAE(InstanceSparseCodingSAE, base_sae.BaseSAE):
    """
    Matching Pursuit Sparse Autoencoder.

    Implements a greedy iterative algorithm that selects features one at a time
    based on maximum correlation with the residual, for S steps.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        s: int,  # number of matching pursuit steps
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)

        assert isinstance(s, int) and s > 0
        self.register_buffer("S", torch.tensor(s, dtype=torch.int, device=device))

    def encode(self, x: torch.Tensor):
        """
        Encode using matching pursuit algorithm.

        Args:
            x: Input activations of shape (batch, seq_len, d_in)

        Returns:
            Sparse codes of shape (batch, seq_len, d_sae)
        """
        batch_size, seq_len, d_in = x.shape

        # Initialize residual: r = x - b_dec
        r = x - self.b_dec  # (batch, seq_len, d_in)

        # Initialize sparse code
        z = torch.zeros(
            batch_size, seq_len, self.W_dec.shape[0],
            device=x.device, dtype=x.dtype
        )

        # Matching pursuit iterations
        for t in range(self.S.item()):
            # Compute correlations: r @ W_dec^T
            correlations = r @ self.W_dec.T  # (batch, seq_len, d_sae)

            # Find feature with maximum absolute correlation per position
            j = torch.argmax(correlations.abs(), dim=-1)  # (batch, seq_len)

            # Extract correlation values for selected features
            # We need to gather along the last dimension
            j_expanded = j.unsqueeze(-1)  # (batch, seq_len, 1)
            z_t = torch.gather(correlations, -1, j_expanded).squeeze(-1)  # (batch, seq_len)

            # Update sparse code: accumulate selected features
            # We scatter_add to handle cases where same feature is selected multiple times
            z.scatter_add_(-1, j_expanded, z_t.unsqueeze(-1))

            # Update residual: r = r - z_t * W_dec[j]
            # Need to select the right dictionary elements for each position
            selected_dict = self.W_dec[j]  # (batch, seq_len, d_in)
            r = r - z_t.unsqueeze(-1) * selected_dict

        return z

    def decode(self, feature_acts: torch.Tensor):
        """Standard linear decode."""
        return (feature_acts @ self.W_dec) + self.b_dec


def load_from_sae_lens(
    model_name: str,
    release: str,
    sae_id: str,
    k: int = None,
    s: int = None,
    hook_layer: int = None,
    sae_cls=InstanceHardThresholdingPursuitSAE,
    device: str = "cuda",
):
    from sae_lens import SAE

    sae_lens_sae = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    d_in = sae_lens_sae.W_dec.shape[1]
    d_sae = sae_lens_sae.W_dec.shape[0]

    # Determine which parameter to use based on the SAE class
    if sae_cls == MPSAE:
        if s is None:
            raise ValueError("Parameter 's' (number of MP steps) is required for MPSAE")
        sae = sae_cls(
            d_in, d_sae, s, model_name, hook_layer, device, sae_lens_sae.W_dec.dtype
        )
    else:
        if k is None:
            raise ValueError("Parameter 'k' is required for InstanceHardThresholdingPursuitSAE")
        sae = sae_cls(
            d_in, d_sae, k, model_name, hook_layer, device, sae_lens_sae.W_dec.dtype
        )

    sae.W_dec.data = sae_lens_sae.W_dec.to(device)
    sae.b_dec.data = sae_lens_sae.b_dec.to(device)
    sae.b_enc.data = sae_lens_sae.b_enc.to(device)
    return sae
