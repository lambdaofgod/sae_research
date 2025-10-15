"""Reusable PyTorch modules for sparse coding operations."""

import torch
import torch.nn as nn


class HardThresholdingSupportSelector(nn.Module):
    """
    Compute support set via instance-level hard thresholding.

    Takes top-k features per position along the feature dimension,
    then returns the union of all selected features across positions.
    This gives per-instance sparsity rather than per-token sparsity.
    """

    def __init__(self):
        super().__init__()

    def forward(self, feature_acts: torch.Tensor, k: int) -> torch.Tensor:
        """
        Find support indices by taking top-k per position, then union.

        Args:
            feature_acts: Feature activations of shape (seq_len, d_sae)
            k: Number of features to select per position

        Returns:
            support: Indices of selected features (1D tensor)
        """
        # Take top-k features per position along the d_sae dimension
        # This gives (seq_len, k)
        top_indices = torch.topk(feature_acts.abs(), k, dim=-1).indices

        # Flatten and get unique feature indices
        # This is the union of all top-k selections across positions
        support = top_indices.flatten().unique()

        return support


class HardThresholding(nn.Module):
    """
    Apply hard thresholding by keeping only top-k values per position.

    Zeros out all values except the top-k (by absolute value) per position
    along the feature dimension.
    """

    def __init__(self, k: int, device: str):
        self.register_buffer("k", torch.tensor(k, dtype=torch.int, device=device))
        super().__init__()

    def forward(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Zero out all but top-k features per position.

        Args:
            feature_acts: Feature activations of shape (seq_len, d_sae)
            k: Number of features to keep per position

        Returns:
            Thresholded activations of shape (seq_len, d_sae)
        """
        # Take top-k features per position along the d_sae dimension
        # This gives (seq_len, k)
        top_indices = torch.topk(feature_acts.abs(), self.k.item(), dim=-1).indices

        thresholded_feature_acts = torch.scatter(
            torch.zeros_like(feature_acts),
            -1,
            top_indices,
            feature_acts.gather(-1, top_indices),
        )
        return thresholded_feature_acts


class SparseLeastSquares(nn.Module):
    """
    Solve sparse least squares on a given support set.

    Fits a linear model using only the features specified by the support indices,
    solving min ||dictionary[support].T @ x - targets||^2 for x.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, dictionary: torch.Tensor, targets: torch.Tensor, support: torch.Tensor
    ) -> torch.Tensor:
        """
        Solve least squares: dictionary[support].T @ x = targets

        Args:
            dictionary: Dictionary matrix of shape (d_sae, d_in)
            targets: Target activations of shape (seq_len, d_in)
            support: Indices of active features (1D tensor)

        Returns:
            x: Sparse coefficients of shape (seq_len, d_sae)
        """
        # Select only the active dictionary elements
        A_in = dictionary[support].float().T  # (d_in, len(support))
        Y_in = targets.float().T  # (d_in, seq_len)

        # Solve least squares
        coefficients = torch.linalg.lstsq(
            A_in, Y_in
        ).solution.T  # (seq_len, len(support))

        # Create sparse output
        x = torch.zeros(
            targets.shape[0],
            dictionary.shape[0],
            device=targets.device,
            dtype=targets.dtype,
        )
        x[:, support] = coefficients.to(targets.dtype)

        return x
