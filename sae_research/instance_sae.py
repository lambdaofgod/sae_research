import json
import torch
import torch.nn as nn
from typing import Union

import sae_bench.custom_saes.base_sae as base_sae
from huggingface_hub import hf_hub_download
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


def load_dictionary_learning_instance_sae(
    repo_id: str,
    filename: str,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer: int | None = None,
    local_dir: str = "downloaded_saes",
) -> Union[InstanceHardThresholdingPursuitSAE, MPSAE]:
    """
    Load an instance SAE (IHTP or MPSAE) from HuggingFace hub.

    This function is analogous to load_dictionary_learning_batch_topk_sae
    but handles instance-level SAE architectures.
    """
    assert "ae.pt" in filename

    # Download the model parameters
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
        local_dir=local_dir,
    )

    pt_params = torch.load(path_to_params, map_location=torch.device("cpu"))

    # Download the config
    config_filename = filename.replace("ae.pt", "config.json")
    path_to_config = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
        force_download=False,
        local_dir=local_dir,
    )

    with open(path_to_config) as f:
        config = json.load(f)

    # Validate/extract layer
    if layer is not None:
        assert layer == config["trainer"]["layer"]
    else:
        layer = config["trainer"]["layer"]

    # Validate model name
    assert model_name in config["trainer"]["lm_name"]

    # Determine SAE type from trainer class and get appropriate parameter
    trainer_class = config["trainer"]["trainer_class"]

    # Print original keys for debugging
    print("Original keys in state_dict:", pt_params.keys())

    # Map old keys to new keys
    key_mapping = {
        "encoder.weight": "W_enc",
        "decoder.weight": "W_dec",
        "encoder.bias": "b_enc",
        "bias": "b_dec",
    }

    # Create a new dictionary with renamed keys
    renamed_params = {key_mapping.get(k, k): v for k, v in pt_params.items()}

    # Transpose weight matrices (due to nn.Linear convention)
    renamed_params["W_enc"] = renamed_params["W_enc"].T
    renamed_params["W_dec"] = renamed_params["W_dec"].T

    # Print renamed keys for debugging
    print("Renamed keys in state_dict:", renamed_params.keys())

    # Create appropriate SAE instance based on trainer class
    d_in = renamed_params["b_dec"].shape[0]
    d_sae = renamed_params["b_enc"].shape[0]

    if trainer_class == "InstanceHardThresholdingPursuitTrainer":
        # IHTP SAE uses k parameter
        k = config["trainer"]["k"]
        sae = InstanceHardThresholdingPursuitSAE(
            d_in=d_in,
            d_sae=d_sae,
            k=k,
            model_name=model_name,
            hook_layer=layer,  # type: ignore
            device=device,
            dtype=dtype,
        )
        # Add k to renamed_params if it's in the config but not the state dict
        if "k" not in renamed_params and "k" in config["trainer"]:
            renamed_params["k"] = torch.tensor(k, dtype=torch.int, device=device)
        sae.cfg.architecture = "instance_hard_thresholding_pursuit"

    elif trainer_class == "MPSAETrainer":
        # MPSAE uses s parameter (number of matching pursuit steps)
        s = config["trainer"]["s"]
        sae = MPSAE(
            d_in=d_in,
            d_sae=d_sae,
            s=s,
            model_name=model_name,
            hook_layer=layer,  # type: ignore
            device=device,
            dtype=dtype,
        )
        # Add S to renamed_params if it's in the config but not the state dict
        if "S" not in renamed_params and "s" in config["trainer"]:
            renamed_params["S"] = torch.tensor(s, dtype=torch.int, device=device)
        sae.cfg.architecture = "mp_sae"

    else:
        raise ValueError(f"Unknown trainer class: {trainer_class}")

    # Load the state dict
    sae.load_state_dict(renamed_params, strict=False)  # type: ignore

    # Move to specified device and dtype
    sae.to(device=device, dtype=dtype)

    # Validate dimensions
    d_sae_actual, d_in_actual = sae.W_dec.data.shape
    assert d_sae_actual >= d_in_actual, f"d_sae ({d_sae_actual}) should be >= d_in ({d_in_actual})"

    # Check decoder normalization
    normalized = sae.check_decoder_norms()
    if not normalized:
        raise ValueError("Decoder vectors are not normalized. Please normalize them")

    return sae
