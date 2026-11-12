import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
import numba
import torch
from tqdm import tqdm
from sae_research.feature_families.utils import perform_spectral_clustering


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
    support_out, coefs_out = solve_pursuit_batch(
        X_batch, X, correlations, k, use_omp, i_start
    )

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


def _default_device():
    """Fastest available torch device: CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _torch_masked_correlations(R, Xd, exclude):
    """|R @ Xd.T| with columns `exclude` (B, m) set to -1 so argmax skips them."""
    corr = (R @ Xd.T).abs_()
    corr.scatter_(1, exclude, -1.0)
    return corr


def _torch_solve_coefs(A, xb):
    """Batched OMP least squares: argmin_c ||xb - c @ A|| for A (B, t, d), xb (B, d).

    The (t, t) normal equations are formed on the device and solved on CPU with
    gelsd (min-norm, handles singular Gram from near-duplicate atoms).
    Singular values below float32 precision are treated as zero.
    """
    G = A @ A.transpose(1, 2)
    r = A @ xb.unsqueeze(-1)
    rcond = torch.finfo(torch.float32).eps * A.shape[1]
    c = torch.linalg.lstsq(G.cpu(), r.cpu(), rcond=rcond, driver="gelsd").solution
    return c.squeeze(-1).to(A.device)


def _torch_pursuit_batch(Xb, Xd, self_idx, k):
    """OMP for a whole batch at once; returns support (B, k) and coefs (B, k)."""
    B = Xb.shape[0]
    support = torch.empty((B, k), dtype=torch.long, device=Xb.device)
    corr = _torch_masked_correlations(Xb, Xd, self_idx)
    for t in range(k):
        support[:, t] = corr.argmax(dim=1)
        A = Xd[support[:, : t + 1]]
        coefs = _torch_solve_coefs(A, Xb)
        if t < k - 1:
            R = Xb - torch.einsum("bt,btd->bd", coefs, A)
            exclude = torch.cat([support[:, : t + 1], self_idx], dim=1)
            corr = _torch_masked_correlations(R, Xd, exclude)
    return support, coefs


def compute_sparse_coefficients_torch(X, k, batch_size, device=None):
    """OMP sparse self-representation coefficients, batched in torch.

    Same algorithm as `compute_sparse_coefficients(..., use_omp=True)`: X is
    assumed row-normalized, each row gets exactly k coefficients on other rows.
    Everything is computed in float32; other input dtypes are cast.
    """
    n_samples = X.shape[0]
    if k >= n_samples:
        raise ValueError(
            f"k={k} must be smaller than n_samples={n_samples} "
            "(each sample needs k other samples to pick from)"
        )
    if device is None:
        device = _default_device()
    Xd = torch.as_tensor(np.ascontiguousarray(X, dtype=np.float32), device=device)

    rows = np.repeat(np.arange(n_samples, dtype=np.int64), k)
    cols = np.empty(n_samples * k, dtype=np.int64)
    data = np.empty(n_samples * k, dtype=np.float32)

    for i_start in tqdm(range(0, n_samples, batch_size), desc="SSC batches"):
        i_end = min(i_start + batch_size, n_samples)
        self_idx = torch.arange(i_start, i_end, device=device).unsqueeze(1)
        support, coefs = _torch_pursuit_batch(Xd[i_start:i_end], Xd, self_idx, k)
        cols[i_start * k : i_end * k] = support.cpu().numpy().ravel()
        data[i_start * k : i_end * k] = coefs.cpu().numpy().ravel()

    return sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))


@numba.jit(nopython=True)
def solve_pursuit_batch(X_batch, Dictionary, correlations, k, use_omp=False, i_start=0):
    """Matching Pursuit (use_omp=False) or Orthogonal MP (use_omp=True).

    X_batch[b] is assumed to be Dictionary[i_start + b]; that atom is excluded
    from its own representation at every iteration (SSC constraint C_ii = 0).
    """
    batch_size = X_batch.shape[0]

    support_out = np.zeros((batch_size, k), dtype=np.int64)
    coefs_out = np.zeros((batch_size, k), dtype=np.float64)

    for b_idx in range(batch_size):
        x_original = X_batch[b_idx].copy()
        y = x_original.copy()
        self_idx = i_start + b_idx
        corr = correlations[b_idx].copy()
        corr[self_idx] = 0

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
            corr[self_idx] = 0

    return support_out, coefs_out


class SparseSubspaceClusteringOMP:
    """Sparse Subspace Clustering using Matching Pursuit or OMP (sklearn-style API).

    OMP (use_omp=True) runs in float32 torch on `device` (None: CUDA, then MPS,
    then CPU). MP (use_omp=False) runs on the numba implementation.
    """

    def __init__(self, n_clusters, k=10, batch_size=1000, use_omp=True, device=None):
        self.n_clusters = n_clusters
        self.k = k
        self.batch_size = batch_size
        self.use_omp = use_omp
        self.device = device
        self.labels_ = None
        self._coefs = None

    def fit(self, X):
        # Step 1: Normalize
        X_norm = normalize(X, axis=1)

        # Step 2: Compute Sparse Coefficients
        if self.use_omp:
            self._coefs = compute_sparse_coefficients_torch(
                X_norm, self.k, self.batch_size, self.device
            )
        else:
            self._coefs = compute_sparse_coefficients(
                X_norm, self.k, self.batch_size, use_omp=False
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
            use_omp=self.use_omp,
            device=self.device,
        )
        new_instance._coefs = self._coefs
        new_instance.labels_ = perform_spectral_clustering(self._coefs, new_n_clusters)
        return new_instance

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
