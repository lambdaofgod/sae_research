from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict, model_validator


class SAEFeatures(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    features: np.ndarray  # shape: (n_features, d_model)
    labels: pd.DataFrame  # columns: feature_index, label

    @property
    def n_features(self) -> int:
        return self.features.shape[0]

    @property
    def d_model(self) -> int:
        return self.features.shape[1]

    @model_validator(mode="after")
    def validate_labels_match_features(self) -> "SAEFeatures":
        if len(self.labels) != self.n_features:
            raise ValueError(
                f"Number of labels ({len(self.labels)}) does not match "
                f"number of features ({self.n_features})"
            )
        return self

    @classmethod
    def from_goodfire_hf(
        cls,
        sae_name: str,
        labels_path: str | Path,
        min_norm: float = 1e-3,
    ) -> "SAEFeatures":
        """Load SAE features from Goodfire HuggingFace repo.

        Args:
            sae_name: Name of SAE (e.g. "Llama-3.1-8B-Instruct-SAE-l19")
            labels_path: Path to CSV with feature labels (columns: feature_index, label)
            min_norm: Minimum L2 norm to keep feature (filters dead features)
        """
        file_path = hf_hub_download(
            repo_id=f"Goodfire/{sae_name}",
            filename=f"{sae_name}.pth",
            repo_type="model",
        )
        sae_params = torch.load(file_path, weights_only=True)
        decoder_weights = sae_params["decoder_linear.weight"]  # (d_model, n_features)

        features_raw = decoder_weights.T.cpu().numpy()  # (n_features, d_model)

        # Filter dead features
        norms = np.linalg.norm(features_raw, axis=1)
        mask = norms >= min_norm
        features = features_raw[mask]

        # Load labels (already pre-filtered in CSV)
        labels = pd.read_csv(labels_path)

        return cls(features=features, labels=labels)


class GroupedSAEFeatures(BaseModel, SAEFeatures):
    group_labels: np.ndarray  # shape: (n_features,)

    @model_validator(mode="after")
    def validate_cluster_labels_match_features(self):
        if len(self.cluster_labels) != self.n_features:
            raise ValueError(
                f"Number of cluster labels ({len(self.cluster_labels)}) does not match "
                f"number of features ({self.n_features})"
            )
        return self
