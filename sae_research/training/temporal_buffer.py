"""
Temporal S3 activation buffer for temporal SAE training.

Migrated from forks/temporal-saes/src/temporal_buffer.py (TemporalS3Buffer only).
Pairs each token at position i with the token at position i-1 within the same
sequence, producing (B*(L-1)) pairs per S3 file.

Yields batches of shape [sae_batch_size, 2, d_model] where:
  - [:,0,:] = current token activations
  - [:,1,:] = prior (immediately preceding) token activations
"""

import einops
import torch as t

from dictionary_learning.activault_s3_buffer import (
    ActivaultS3ActivationBuffer,
    S3RCache,
)


class TemporalS3Buffer(ActivaultS3ActivationBuffer):
    """
    Subclass of ActivaultS3ActivationBuffer that yields [batch, 2, d_model] tensors
    for temporal SAE training. Pairs each token at position i with the token at
    position i-1 within the same sequence, producing (B*(L-1)) pairs per S3 file.

    The S3RCache must be created with return_ids=True to preserve sequence order;
    a ValueError is raised at construction time otherwise.
    """

    def __init__(self, cache: S3RCache, *args, **kwargs):
        if not cache.return_ids:
            raise ValueError(
                "TemporalS3Buffer requires the S3RCache to be created with "
                "return_ids=True to preserve sequence order for temporal pairing."
            )
        super().__init__(cache, *args, **kwargs)

    def refresh(self):
        try:
            next_batch = next(self.cache)
        except StopIteration:
            self.states = None
            self.read_mask = None
            return

        states = next_batch["states"].to(self.device)  # [B, L, D]
        # Build adjacent pairs: current = position i, previous = position i-1.
        pairs = t.stack([states[:, 1:], states[:, :-1]], dim=2)  # [B, L-1, 2, D]
        flat_pairs = einops.rearrange(pairs, "b l p d -> (b l) p d").contiguous()

        self.states = flat_pairs
        self.read_mask = t.zeros(flat_pairs.shape[0], dtype=t.bool, device=self.device)
