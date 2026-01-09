#!/usr/bin/env python
"""Run Cosine Similarity Clustering on SAE features."""
from pathlib import Path

import dvc.api
import fire

from sae_research.feature_families import CosineSimilarityClustering
from sae_research.sae_features import SAEFeatures


def main(threshold: str) -> None:
    """Run CSC with a given threshold.

    Args:
        threshold: Threshold value as scientific notation string (e.g., "5e-2")
    """
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
    csc = CosineSimilarityClustering(
        threshold=threshold_float,
        n_clusters=csc_params["n_clusters"],
        batch_size=csc_params["batch_size"],
    )
    csc.fit(sae_features.features)

    # Save results
    output_dir = Path("results/csc")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"csc_T{threshold}.csv"
    result_df = sae_features.labels.copy()
    result_df["cluster"] = csc.labels_
    result_df.to_csv(output_path, index=False)

    print(f"Saved {len(result_df)} features to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
