#!/usr/bin/env python
"""Compute SSC OMP coefficients for SAE features."""
import pickle
from pathlib import Path

import dvc.api
import fire

from sae_research.feature_families import SparseSubspaceClusteringOMP
from sae_research.sae_features import SAEFeatures


def main(k: int) -> None:
    """Compute SSC coefficients with given k.

    Args:
        k: Sparsity level (number of neighbors in OMP)
    """
    params = dvc.api.params_show()
    data_params = params["data"]
    ssc_params = params["ssc"]

    # Load SAE features
    sae_features = SAEFeatures.from_goodfire_hf(
        sae_name=data_params["sae_name"],
        labels_path=data_params["labels_path"],
        min_norm=data_params["min_norm"],
    )

    # Compute SSC coefficients (use n_clusters=1 as placeholder, we only need coefs)
    ssc = SparseSubspaceClusteringOMP(
        n_clusters=1,
        k=k,
        batch_size=ssc_params["batch_size"],
        use_omp=True,
    )
    ssc.fit(sae_features.features)

    # Save coefficients
    output_dir = Path("data/ssc_coefs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"coefs_k{k}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(ssc._coefs, f)

    print(f"Saved coefficients to {output_path}")
    print(f"Coefficients shape: {ssc._coefs.shape}")


if __name__ == "__main__":
    fire.Fire(main)
