import numpy as np
from sklearn.cluster import KMeans
from sknetwork.embedding import Spectral


def perform_spectral_clustering(C_sparse, n_clusters, n_init=3, random_state=None):
    """Spectral clustering of the affinity W = |C| + |C|^T.

    Embedding: the n_clusters leading random-walk eigenvectors of W (scikit-network,
    scipy eigsh on the normalized Laplacian), then k-means with k-means++ init.
    Replaces sklearn SpectralClustering(eigen_solver="amg"), whose lobpcg did not
    converge and was about 2.6x slower at N=45k, k=300; n_init=3 matched n_init=10
    on the synthetic benchmark (dev/spectral_clustering/).
    """
    W = np.abs(C_sparse) + np.abs(C_sparse).T
    emb = Spectral(
        n_components=n_clusters, decomposition="rw", normalized=False
    ).fit_transform(W)
    return KMeans(
        n_clusters=n_clusters, n_init=n_init, random_state=random_state
    ).fit_predict(emb)
