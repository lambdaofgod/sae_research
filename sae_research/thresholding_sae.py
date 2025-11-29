"""
Implements the SAE training scheme from https://arxiv.org/abs/2406.04093.
Significant portions of this code have been copied from https://github.com/EleutherAI/sae/blob/main/sae
"""

import torch as t
import torch.nn as nn
from typing import Optional, Dict, Tuple, Union
from jaxtyping import Float, Int

from dictionary_learning.dictionary import Dictionary
from dictionary_learning.trainers.trainer import (
    set_decoder_norm_to_unit_norm,
)

# SAE inputs are pre-flattened: (batch, seq, activation_dim) -> (n_tokens, activation_dim)
# where n_tokens = batch * sequence_length
# The "B" in variable names (_BD, _BF, _BK) represents n_tokens


@t.no_grad()
def geometric_median(
    points: Float[t.Tensor, "n_points dim"], max_iter: int = 100, tol: float = 1e-5
) -> Float[t.Tensor, "dim"]:
    """Compute the geometric median `points`. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    prev = t.zeros_like(guess)

    # Weights for iteratively reweighted least squares
    weights = t.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess

        # Compute the weights
        weights = 1 / t.norm(points - guess, dim=1)

        # Normalize the weights
        weights /= weights.sum()

        # Compute the new geometric median
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        # Early stopping condition
        if t.norm(guess - prev) < tol:
            break

    return guess


class ThresholdingAutoEncoderTopK(Dictionary, nn.Module):
    """
    Hard thresholding top-k autoencoder using a single decoder matrix.

    This implementation uses hard thresholding based on absolute values of activations,
    selecting the top-k features with largest absolute values while preserving their signs.
    Uses only a single weight matrix W_decoder for both encoding and decoding.

    Encoding: acts = (x - b_dec) @ W_decoder
    Selection: top-k based on |acts|, preserving signs
    """

    def __init__(self, activation_dim: int, dict_size: int, k: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        self.register_buffer("k", t.tensor(k, dtype=t.int))
        self.register_buffer("threshold", t.tensor(-1.0, dtype=t.float32))

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, activation_dim, dict_size
        )

        self.b_dec = nn.Parameter(t.zeros(activation_dim))

    def encode(
        self,
        x: Float[t.Tensor, "n_tokens activation_dim"],
        return_topk: bool = False,
        use_threshold: bool = False,
    ) -> Union[
        Float[t.Tensor, "n_tokens dict_size"],
        Tuple[
            Float[t.Tensor, "n_tokens dict_size"],  # encoded_acts
            Float[t.Tensor, "n_tokens k"],  # top_acts
            Int[t.Tensor, "n_tokens k"],  # top_indices
            Float[t.Tensor, "n_tokens dict_size"],  # feat_acts
        ],
    ]:
        # Hard thresholding: acts = (x - b_dec) @ W_decoder.T
        # Note: decoder.weight is stored as (activation_dim, dict_size) in nn.Linear
        # So we need to use decoder.weight without transpose
        feat_acts_BF = (x - self.b_dec) @ self.decoder.weight

        if use_threshold:
            # Apply threshold based on absolute values
            encoded_acts_BF = feat_acts_BF * (t.abs(feat_acts_BF) > self.threshold)
            if return_topk:
                # Get top-k based on absolute values
                abs_acts_BF = t.abs(feat_acts_BF)
                post_topk = abs_acts_BF.topk(self.k, sorted=False, dim=-1)
                top_indices_BK = post_topk.indices
                # Get the actual values (with signs) at the top-k indices
                top_acts_BK = feat_acts_BF.gather(-1, top_indices_BK)
                return (
                    encoded_acts_BF,
                    top_acts_BK,
                    top_indices_BK,
                    feat_acts_BF,
                )
            else:
                return encoded_acts_BF

        # Get top-k based on absolute values
        abs_acts_BF = t.abs(feat_acts_BF)
        post_topk = abs_acts_BF.topk(self.k, sorted=False, dim=-1)
        top_indices_BK = post_topk.indices

        # Get the actual values (with signs) at the top-k indices
        top_acts_BK = feat_acts_BF.gather(-1, top_indices_BK)

        buffer_BF = t.zeros_like(feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(
            dim=-1, index=top_indices_BK, src=top_acts_BK
        )

        if return_topk:
            return encoded_acts_BF, top_acts_BK, top_indices_BK, feat_acts_BF
        else:
            return encoded_acts_BF

    def decode(
        self, x: Float[t.Tensor, "n_tokens dict_size"]
    ) -> Float[t.Tensor, "n_tokens activation_dim"]:
        return self.decoder(x) + self.b_dec

    def forward(
        self,
        x: Float[t.Tensor, "n_tokens activation_dim"],
        output_features: bool = False,
    ) -> Union[
        Float[t.Tensor, "n_tokens activation_dim"],
        Tuple[
            Float[t.Tensor, "n_tokens activation_dim"],
            Float[t.Tensor, "n_tokens dict_size"],
        ],
    ]:
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)
        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    def scale_biases(self, scale: float) -> None:
        self.b_dec.data *= scale
        if self.threshold >= 0:
            self.threshold *= scale

    @classmethod
    def from_pretrained(
        cls, path: str, k: Optional[int] = None, device: Optional[str] = None
    ) -> "ThresholdingAutoEncoderTopK":
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)

        # Handle backward compatibility - check for encoder weights in old format
        if "encoder.weight" in state_dict:
            dict_size, activation_dim = state_dict["encoder.weight"].shape
            # Remove encoder weights from state dict for new architecture
            del state_dict["encoder.weight"]
            if "encoder.bias" in state_dict:
                del state_dict["encoder.bias"]
        else:
            # New format - get dimensions from decoder
            activation_dim, dict_size = state_dict["decoder.weight"].shape

        if k is None:
            k = state_dict["k"].item()
        elif "k" in state_dict and k != state_dict["k"].item():
            raise ValueError(f"k={k} != {state_dict['k'].item()}=state_dict['k']")

        autoencoder = ThresholdingAutoEncoderTopK(activation_dim, dict_size, k)
        autoencoder.load_state_dict(
            state_dict, strict=False
        )  # Allow missing encoder keys
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class NestedThresholdingAutoEncoderTopK(ThresholdingAutoEncoderTopK):
    """
    Nested hard thresholding top-k autoencoder using a single decoder matrix.

    This implementation uses hard thresholding based on absolute values of activations,
    selecting different top-k values for nested models with varying sparsity levels.
    Uses only a single weight matrix W_decoder for both encoding and decoding.

    Encoding: acts = (x - b_dec) @ W_decoder
    Selection: top-k based on |acts| for multiple k values, preserving signs
    """

    def __init__(self, activation_dim: int, dict_size: int, k_values: list[int]):
        # Validate and sort k values
        assert all(
            isinstance(k, int) and k > 0 for k in k_values
        ), "All k values must be positive integers"
        self.k_values = sorted(k_values)  # Ensure ascending order
        self.max_k = max(self.k_values)

        # Call parent init with max_k
        super().__init__(activation_dim, dict_size, self.max_k)

        # Store k_values as buffer for device handling
        self.register_buffer("k_tensor", t.tensor(self.k_values, dtype=t.int))

    def encode(
        self, x: Float[t.Tensor, "n_tokens activation_dim"]
    ) -> Float[t.Tensor, "n_tokens dict_size"]:
        """
        Standard encode for backward compatibility - returns encoding for max_k.

        Args:
            x: Input tensor of shape (n_tokens, activation_dim)

        Returns:
            Tensor of encoded_acts for max_k
        """
        # Hard thresholding: acts = (x - b_dec) @ W_decoder.T
        feat_acts_BF = (x - self.b_dec) @ self.decoder.weight

        # Get top-k based on absolute values for max_k
        abs_acts_BF = t.abs(feat_acts_BF)
        post_topk = abs_acts_BF.topk(self.max_k, sorted=True, dim=-1)
        top_indices_BK = post_topk.indices

        # Get the actual values (with signs) at the top-k indices
        top_acts_BK = feat_acts_BF.gather(-1, top_indices_BK)

        # Return max_k encoding
        buffer_BF = t.zeros_like(feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(
            dim=-1, index=top_indices_BK, src=top_acts_BK
        )
        return encoded_acts_BF

    def encode_nested(
        self, x: Float[t.Tensor, "n_tokens activation_dim"]
    ) -> Dict[int, Float[t.Tensor, "n_tokens dict_size"]]:
        """
        Encode with all nested k values.

        Args:
            x: Input tensor of shape (n_tokens, activation_dim)

        Returns:
            Dict[int, Tensor] mapping k -> encoded_acts
        """
        # Hard thresholding: acts = (x - b_dec) @ W_decoder.T
        feat_acts_BF = (x - self.b_dec) @ self.decoder.weight

        # Get top-k based on absolute values for max_k
        abs_acts_BF = t.abs(feat_acts_BF)
        post_topk = abs_acts_BF.topk(self.max_k, sorted=True, dim=-1)
        top_indices_BK = post_topk.indices

        # Get the actual values (with signs) at the top-k indices
        top_acts_BK = feat_acts_BF.gather(-1, top_indices_BK)

        # Create nested encodings for each k value
        nested_encodings = {}
        for k in self.k_values:
            # Take first k features (since sorted=True)
            k_indices = top_indices_BK[:, :k]
            k_acts = top_acts_BK[:, :k]

            buffer_BF = t.zeros_like(feat_acts_BF)
            encoded_acts_k = buffer_BF.scatter_(dim=-1, index=k_indices, src=k_acts)
            nested_encodings[k] = encoded_acts_k

        return nested_encodings

    def encode_with_info(self, x: Float[t.Tensor, "n_tokens activation_dim"]) -> Tuple[
        Dict[int, Float[t.Tensor, "n_tokens dict_size"]],
        Float[t.Tensor, "n_tokens dict_size"],
    ]:
        """
        Encode with additional info for training.

        Returns:
            (nested_encodings, feat_acts_BF) for auxiliary loss computation
        """
        feat_acts_BF = (x - self.b_dec) @ self.decoder.weight
        nested_encodings = self.encode_nested(x)
        return nested_encodings, feat_acts_BF

    def scale_biases(self, scale: float) -> None:
        self.b_dec.data *= scale

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        k_values: Optional[list[int]] = None,
        device: Optional[str] = None,
    ) -> "NestedThresholdingAutoEncoderTopK":
        """
        Load a pretrained nested autoencoder from a file.
        """
        checkpoint = t.load(path, map_location='cpu')

        # Handle both raw state_dict and training checkpoint formats
        if 'ae' in checkpoint:
            # This is a training checkpoint, extract the model state_dict
            state_dict = checkpoint['ae']
        else:
            # This is already a state_dict
            state_dict = checkpoint

        # Get dimensions from decoder - try multiple possible keys
        decoder_weight_key = None
        for key in ["decoder.weight", "decoder._weight"]:
            if key in state_dict:
                decoder_weight_key = key
                break

        if decoder_weight_key is None:
            available_keys = [k for k in state_dict.keys() if "decoder" in k]
            raise KeyError(
                f"Could not find decoder weight in state dict. "
                f"Available decoder keys: {available_keys}"
            )

        activation_dim, dict_size = state_dict[decoder_weight_key].shape

        if k_values is None:
            k_values = state_dict["k_tensor"].tolist()
        elif "k_tensor" in state_dict:
            saved_k = state_dict["k_tensor"].tolist()
            if sorted(k_values) != sorted(saved_k):
                raise ValueError(f"k_values={k_values} != saved k_values={saved_k}")

        autoencoder = NestedThresholdingAutoEncoderTopK(
            activation_dim, dict_size, k_values
        )

        # Use strict=False to allow missing keys (k, threshold) from old checkpoints
        autoencoder.load_state_dict(state_dict, strict=False)

        if device is not None:
            autoencoder.to(device)
        return autoencoder
