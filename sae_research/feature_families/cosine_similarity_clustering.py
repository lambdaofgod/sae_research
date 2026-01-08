"""
This is code for feature clustering based on https://arxiv.org/pdf/2405.14860

We now argue that SAEs can discover irreducible multi-dimensional features by clustering D. We
will consider a simple form of clustering: build a complete graph on D with edge weights equal to
the cosine similarity between dictionary elements, prune all edges below a threshold T , and then set
the clusters equal to the connected components of the graph. If we now consider the spaces spanned
by each cluster, they will be approximately T -orthogonal by construction, since their basis vectors
are all T -orthogonal.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import tqdm

from sae_research.feature_families.utils import perform_spectral_clustering


class CosineSimilarityClustering:
    """Clustering based on cosine similarity thresholding (sklearn-style API).

    Builds a complete graph with cosine similarity edge weights,
    prunes edges below threshold T, and returns connected components.
    """

    def __init__(self, threshold, batch_size=1000, n_clusters=None):
        self.threshold = threshold
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self.labels_ = None
        self.n_clusters_ = None
        self._thresholded_similarities = None

    def _compute_thresholded_similarity_matrix(self, X):
        """Compute cosine similarity in batches, returning sparse matrix with only edges >= threshold."""
        X_norm = normalize(X, axis=1)
        n_samples = X_norm.shape[0]

        similarity_matrix = sparse.lil_matrix((n_samples, n_samples))

        for i_start in tqdm.trange(0, n_samples, self.batch_size):
            i_end = min(i_start + self.batch_size, n_samples)
            batch = X_norm[i_start:i_end]

            similarities = batch @ X_norm.T
            similarities[similarities < self.threshold] = 0
            batch_sparse = sparse.csr_matrix(similarities)
            similarity_matrix[i_start:i_end, :] = batch_sparse

        return similarity_matrix.tocsr()

    def _cluster_with_connected_components(self, similarity_matrix):
        """Find connected components in the graph defined by similarity matrix."""
        return connected_components(
            similarity_matrix, directed=False, return_labels=True
        )

    def fit(self, X):
        self._thresholded_similarities = self._compute_thresholded_similarity_matrix(X)
        if self.n_clusters is None:
            self.n_clusters_, self.labels_ = self._cluster_with_connected_components(
                self._thresholded_similarities
            )
        else:
            self.labels_ = perform_spectral_clustering(
                self._thresholded_similarities, self.n_clusters
            )
            self.n_clusters_ = self.n_clusters
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
