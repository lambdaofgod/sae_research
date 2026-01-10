#!/usr/bin/env python
"""Run Cosine Similarity Clustering on SAE features."""
from pathlib import Path

import dvc.api
import fire

from sae_research.feature_families import CosineSimilarityClustering
from sae_research.sae_features import SAEFeatures


def main(threshold: str, mode: str) -> None:
    """Run CSC with a given threshold and mode.

    Args:
        threshold: Threshold value as string (e.g., "0.05")
        mode: Clustering mode - "cc" for connected components, "spectral" for spectral clustering
    """
    if mode not in ("cc", "spectral"):
        raise ValueError(f"mode must be 'cc' or 'spectral', got '{mode}'")

    params = dvc.api.params_show()
    data_params = params["data"]
    csc_params = params["csc"]

    # Load SAE features
    sae_features = SAEFeatures.from_goodfire_hf(
        sae_name=data_params["sae_name"],
        labels_path=data_params["labels_path"],
        min_norm=data_params["min_norm"],
    )

    # Run CSC
    threshold_float = float(threshold)
    n_clusters = csc_params["n_clusters"] if mode == "spectral" else None

    csc = CosineSimilarityClustering(
        threshold=threshold_float,
        n_clusters=n_clusters,
        batch_size=csc_params["batch_size"],
    )
    csc.fit(sae_features.features)

    # Save results
    output_dir = Path("results/csc")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"csc_{mode}_T{threshold}.csv"
    result_df = sae_features.labels.copy()
    result_df["cluster"] = csc.labels_
    result_df.to_csv(output_path, index=False)

    print(f"Mode: {mode}, n_clusters: {csc.n_clusters_}")
    print(f"Saved {len(result_df)} features to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
