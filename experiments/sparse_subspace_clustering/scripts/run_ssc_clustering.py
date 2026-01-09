#!/usr/bin/env python
"""Run spectral clustering on precomputed SSC coefficients."""
import pickle
from pathlib import Path

import dvc.api
import fire

from sae_research.feature_families import SparseSubspaceClusteringOMP
from sae_research.sae_features import SAEFeatures


def main(k: int) -> None:
    """Run spectral clustering on SSC coefficients.

    Args:
        k: Sparsity level used when computing coefficients
    """
    params = dvc.api.params_show()
    data_params = params["data"]
    ssc_params = params["ssc"]

    # Load coefficients
    coefs_path = Path(f"data/ssc_coefs/coefs_k{k}.pkl")
    with open(coefs_path, "rb") as f:
        coefs = pickle.load(f)

    # Create SSC instance with loaded coefficients
    ssc = SparseSubspaceClusteringOMP(
        n_clusters=ssc_params["n_clusters"],
        k=k,
        batch_size=ssc_params["batch_size"],
        use_omp=True,
    )
    ssc._coefs = coefs
    ssc = ssc.recompute_clustering(n_clusters=ssc_params["n_clusters"])

    # Load labels for output
    sae_features = SAEFeatures.from_goodfire_hf(
        sae_name=data_params["sae_name"],
        labels_path=data_params["labels_path"],
        min_norm=data_params["min_norm"],
    )

    # Save results
    output_dir = Path("results/ssc")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"ssc_k{k}.csv"
    result_df = sae_features.labels.copy()
    result_df["cluster"] = ssc.labels_
    result_df.to_csv(output_path, index=False)

    print(f"Saved {len(result_df)} features to {output_path}")
    print(f"Cluster distribution: {result_df['cluster'].value_counts().describe()}")


if __name__ == "__main__":
    fire.Fire(main)
