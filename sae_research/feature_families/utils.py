import numpy as np
from sklearn.cluster import SpectralClustering


def perform_spectral_clustering(C_sparse, n_clusters):
    # Build Affinity Matrix: W = |C| + |C|^T
    W = np.abs(C_sparse) + np.abs(C_sparse).T

    # amg solver is used because it's the one that works for sparse matrices
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        eigen_solver="amg",
        assign_labels="kmeans",
    )
    return sc.fit_predict(W)
