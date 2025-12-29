#!/usr/bin/env python3

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering
import tqdm


def compute_sparse_coefficients(X, k, batch_size):
    n_samples, n_features = X.shape
    rows, cols, data = [], [], []

    for i_start in tqdm.trange(0, n_samples, batch_size):
        i_end = min(i_start + batch_size, n_samples)
        X_batch = X[i_start:i_end]

        # 1. Batch Correlation: This is the expensive part, now vectorized!
        # Result is (batch_size, n_samples)
        correlations = np.abs(X_batch @ X.T)

        # 2. Mask self-representation (set diagonal-like entries to zero)
        for b_idx in range(i_end - i_start):
            correlations[b_idx, i_start + b_idx] = 0

        # 3. Solve OMP for this batch
        # We still iterate through atoms 1..k, but we do it for the whole batch
        batch_coefs = solve_omp_batch(X_batch, X, correlations, k)

        # 4. Store in sparse format
        for b_idx, (idx_set, c_vals) in enumerate(batch_coefs):
            rows.extend([i_start + b_idx] * len(idx_set))
            cols.extend(idx_set)
            data.extend(c_vals)

    return sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))


def solve_omp_batch(X_batch, Dictionary, correlations, k):
    """
    Refined helper to perform OMP.
    While we still solve individual Least Squares,
    the expensive 'search' for atoms is handled via the correlation matrix.
    """
    results = []
    # For each signal in the batch, we find its k atoms
    for b_idx in range(X_batch.shape[0]):
        y = X_batch[b_idx]
        corr = correlations[b_idx]

        support = []
        for _ in range(k):
            # Pick best atom index
            best_atom_idx = np.argmax(corr)
            support.append(best_atom_idx)
            # Mask this atom so it isn't picked again
            corr[best_atom_idx] = 0

            # Efficient Least Squares for small k
            # Solve: min ||y - Dictionary[support] @ c||
            # Dictionary[support] is (k, D). This is very fast.
            A = Dictionary[support].T
            c, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

            # Residual update (optional but technically required for OMP)
            # In pure SSC-OMP, you can often just pick top-k correlations,
            # but true OMP updates the residual to find the 'orthogonal' next atom.
            residual = y - A @ c
            # Update correlations for the next iteration
            corr = np.abs(residual @ Dictionary.T)
            # Re-mask support and self
            corr[support] = 0
            # (Self-mask is already handled in the outer loop)

        results.append((support, c))
    return results


def perform_spectral_clustering(C_sparse, n_clusters):
    # Build Affinity Matrix: W = |C| + |C|^T
    W = np.abs(C_sparse) + np.abs(C_sparse).T

    # Use ARPACK solver - it never 'densifies' the matrix.
    # It only performs matrix-vector multiplications (W * v),
    # which is O(nnz) - extremely fast for sparse matrices.
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        eigen_solver="arpack",  # Required for sparse efficiency
        assign_labels="kmeans",
    )
    return sc.fit_predict(W)


class SparseSubspaceClusteringOMP:
    """Sparse Subspace Clustering using OMP (sklearn-style API)."""

    def __init__(self, n_clusters, k=10, batch_size=1000):
        self.n_clusters = n_clusters
        self.k = k
        self.batch_size = batch_size
        self.labels_ = None
        self._coefs = None

    def fit(self, X):
        # Step 1: Normalize
        X_norm = normalize(X, axis=1)

        # Step 2: Compute Sparse Coefficients (OMP)
        self._coefs = compute_sparse_coefficients(X_norm, self.k, self.batch_size)

        # Step 3: Spectral Clustering
        self.labels_ = perform_spectral_clustering(self._coefs, self.n_clusters)
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
