#!/usr/bin/env python3

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering
import numba
from tqdm import tqdm


@numba.jit(nopython=True)
def _compute_coefs_and_residual(x_original, residual, Dictionary, support_arr, use_omp):
    """Compute coefficients and residual for pursuit algorithms."""
    if use_omp:
        # OMP: solve lstsq against original signal
        A = Dictionary[support_arr].T
        coefs, _, _, _ = np.linalg.lstsq(A, x_original)
        new_residual = x_original - A @ coefs
    else:
        # MP: compute coefficient against current residual
        atom = Dictionary[support_arr[-1]]
        coef = np.dot(residual, atom)
        coefs = np.array([coef])
        new_residual = residual - coef * atom
    return coefs, new_residual


@numba.jit(nopython=True)
def _compute_batch_coefficients(X, i_start, batch_size, k, use_omp, rows, cols, data):
    """Compute sparse coefficients for a single batch."""
    n_samples = X.shape[0]
    i_end = min(i_start + batch_size, n_samples)
    X_batch = X[i_start:i_end]
    actual_batch_size = i_end - i_start

    # Batch correlation
    correlations = np.abs(X_batch @ X.T)

    # Mask self-representation
    for b_idx in range(actual_batch_size):
        correlations[b_idx, i_start + b_idx] = 0

    # Solve pursuit
    support_out, coefs_out = solve_pursuit_batch(X_batch, X, correlations, k, use_omp)

    # Store in pre-allocated arrays
    for b_idx in range(actual_batch_size):
        sample_idx = i_start + b_idx
        offset = sample_idx * k
        for j in range(k):
            rows[offset + j] = sample_idx
            cols[offset + j] = support_out[b_idx, j]
            data[offset + j] = coefs_out[b_idx, j]


def compute_sparse_coefficients(X, k, batch_size, use_omp=False):
    n_samples = X.shape[0]

    # Pre-allocate: each sample has k entries
    rows = np.zeros(n_samples * k, dtype=np.int64)
    cols = np.zeros(n_samples * k, dtype=np.int64)
    data = np.zeros(n_samples * k, dtype=np.float64)

    batch_starts = range(0, n_samples, batch_size)
    for i_start in tqdm(batch_starts, desc="SSC batches"):
        _compute_batch_coefficients(
            X, i_start, batch_size, k, use_omp, rows, cols, data
        )

    return sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))


@numba.jit(nopython=True)
def solve_pursuit_batch(X_batch, Dictionary, correlations, k, use_omp=False):
    """Matching Pursuit (use_omp=False) or Orthogonal MP (use_omp=True)."""
    batch_size = X_batch.shape[0]

    support_out = np.zeros((batch_size, k), dtype=np.int64)
    coefs_out = np.zeros((batch_size, k), dtype=np.float64)

    for b_idx in range(batch_size):
        x_original = X_batch[b_idx].copy()
        y = x_original.copy()
        corr = correlations[b_idx].copy()

        for i in range(k):
            best_idx = np.argmax(corr)
            support_out[b_idx, i] = best_idx
            corr[best_idx] = 0

            coefs, y = _compute_coefs_and_residual(
                x_original, y, Dictionary, support_out[b_idx, : i + 1], use_omp
            )
            if use_omp:
                coefs_out[b_idx, : i + 1] = coefs
            else:
                coefs_out[b_idx, i] = coefs[-1]

            corr = np.abs(y @ Dictionary.T)
            corr[support_out[b_idx, : i + 1]] = 0

    return support_out, coefs_out


def perform_spectral_clustering(C_sparse, n_clusters):
    # Build Affinity Matrix: W = |C| + |C|^T
    W = np.abs(C_sparse) + np.abs(C_sparse).T

    # Use ARPACK solver - it never 'densifies' the matrix.
    # It only performs matrix-vector multiplications (W * v),
    # which is O(nnz) - extremely fast for sparse matrices.
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        eigen_solver="amg",
        assign_labels="kmeans",
    )
    return sc.fit_predict(W)


class SparseSubspaceClusteringOMP:
    """Sparse Subspace Clustering using Matching Pursuit or OMP (sklearn-style API)."""

    def __init__(self, n_clusters, k=10, batch_size=1000, use_omp=True):
        self.n_clusters = n_clusters
        self.k = k
        self.batch_size = batch_size
        self.use_omp = use_omp
        self.labels_ = None
        self._coefs = None

    def fit(self, X):
        # Step 1: Normalize
        X_norm = normalize(X, axis=1)

        # Step 2: Compute Sparse Coefficients
        self._coefs = compute_sparse_coefficients(
            X_norm, self.k, self.batch_size, self.use_omp
        )

        # Step 3: Spectral Clustering
        self.labels_ = perform_spectral_clustering(self._coefs, self.n_clusters)
        return self

    def recompute_clustering(self, n_clusters=None):
        """Re-run spectral clustering using stored sparse coefficients.

        This method allows re-clustering with a different number of clusters
        without recomputing the expensive OMP sparse representation.
        Returns a new instance without modifying the original.

        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters. If None, uses the current n_clusters value.

        Returns
        -------
        SparseSubspaceClusteringOMP
            New instance with updated labels_ and shared _coefs.

        Raises
        ------
        ValueError
            If fit() has not been called yet (no stored coefficients).
        """
        if self._coefs is None:
            raise ValueError(
                "No sparse coefficients found. Call 'fit' before 'recompute_clustering'."
            )

        new_n_clusters = n_clusters if n_clusters is not None else self.n_clusters

        new_instance = SparseSubspaceClusteringOMP(
            n_clusters=new_n_clusters,
            k=self.k,
            batch_size=self.batch_size,
            use_omp=self.use_omp
        )
        new_instance._coefs = self._coefs
        new_instance.labels_ = perform_spectral_clustering(self._coefs, new_n_clusters)
        return new_instance

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
