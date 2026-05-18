"""Dyadic cells decomposition via METIS over an affinity-weighted kNN graph.

Implements the construction in Allard, Chen & Maggioni (2011),
"Multiscale Geometric Methods for Data Sets II: Geometric Multi-Resolution
Analysis" (arXiv:1105.4924), §2.1 (p. 5). The output is a `Hierarchy`
consumable by `gmra.py` unchanged — same `HierarchyNode` / `Hierarchy` shape
that `tree_adapters.evoc_to_hierarchy` produces.

Pipeline
--------
    X (n × d)
        --pynndescent-->                k-NN graph (idx, dist)
        --gaussian kernel + symmetrize-->   weighted CSR affinity graph
        --recursive METIS bisection-->      dyadic binary tree
        --DFS adapter-->                    Hierarchy

The induced-subgraph extraction inside the recursion is the per-call hot path
at million scale and is JIT-compiled with numba. Affinity-weight evaluation is
left as vectorized numpy; profile and add a numba kernel only if it shows up.

`node_id` convention is `(depth, ordinal_at_depth)` (root = `(0, 0)`),
matching the `tuple[int, int]` shape used by EVoC ids so nothing in `gmra.py`
needs to know which adapter built the tree.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numba
import numpy as np
import pymetis
import pynndescent
from scipy.sparse import csr_matrix

from tree_adapters import Hierarchy, HierarchyNode

logger = logging.getLogger(__name__)


# ---------- 1. kNN affinity graph ----------


def build_knn_affinity_graph(
    X: np.ndarray,
    k: int = 50,
    *,
    random_state: int = 42,
    n_jobs: int = -1,
) -> csr_matrix:
    """Symmetrized affinity-weighted kNN graph (paper §2.1, p. 5).

    Construction:
        eps_i = distance from x_i to its (k//2)-th non-self neighbor
        W_ij  = exp(- ||x_i − x_j||² / (eps_i · eps_j))
        A     = max(W, Wᵀ)   (element-wise; gives the strongest of the two
                              directed weights on each undirected edge)

    Parameters
    ----------
    X : (n, d) float array.
    k : neighbors per point including self. Default 50 (paper).
    random_state : seed forwarded to NNDescent.
    n_jobs : NNDescent worker count.

    Returns
    -------
    csr_matrix, shape (n, n), float32, symmetric, zero diagonal.
    """
    n = X.shape[0]
    if k < 4:
        raise ValueError(
            f"k must be at least 4 so k//2 >= 2 gives a meaningful bandwidth, got {k}"
        )
    logger.info("build_knn_affinity_graph: n=%d d=%d k=%d", n, X.shape[1], k)

    nn = pynndescent.NNDescent(
        X,
        n_neighbors=k,
        metric="euclidean",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    idx, dist = nn.neighbor_graph
    # Convention: idx[i, 0] == i, dist[i, 0] == 0.

    # Bandwidth: distance to (k//2)-th non-self neighbor lives at column k//2.
    eps = dist[:, k // 2].astype(np.float32, copy=False)
    # Duplicate points → eps==0 → division blows up. Floor at tiny.
    eps = np.maximum(eps, np.finfo(np.float32).tiny)

    # Drop the self column.
    nbr_idx = idx[:, 1:]  # (n, k-1) int32
    nbr_dist = dist[:, 1:].astype(np.float32, copy=False)  # (n, k-1)

    rows = np.repeat(np.arange(n, dtype=np.int32), k - 1)
    cols = nbr_idx.ravel().astype(np.int32, copy=False)
    eps_prod = eps[rows] * eps[cols]
    weights = np.exp(-(nbr_dist.ravel() ** 2) / eps_prod).astype(np.float32, copy=False)

    A = csr_matrix((weights, (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)  # symmetrize via element-wise max
    A.setdiag(0)
    A.eliminate_zeros()
    A.sort_indices()
    logger.info("build_knn_affinity_graph: nnz=%d", A.nnz)
    return A.tocsr()


# ---------- 2. METIS recursive bisection ----------


@numba.njit(cache=True)
def _extract_induced_subgraph(
    indptr,  # int32[n+1]
    indices,  # int32[nnz]
    data,  # float32[nnz]
    in_subset,  # bool[n]   — True iff vertex is in the subset
    vertex_relabel,  # int32[n]  — global → local index (only valid where in_subset)
    vertex_subset,  # int32[n_local]  — global vertices in subset, in local order
):
    """CSR of the subgraph induced by `vertex_subset`.

    Two-pass over the parent CSR:
      pass 1: count surviving edges per local vertex → sub_indptr
      pass 2: fill sub_indices / sub_data

    If the input is symmetric, the output is symmetric (a kept edge u→v has its
    twin v→u in the parent CSR and both endpoints lie in the subset).
    """
    n_local = vertex_subset.shape[0]
    sub_indptr = np.zeros(n_local + 1, dtype=np.int64)
    for li in range(n_local):
        gv = vertex_subset[li]
        cnt = 0
        for ei in range(indptr[gv], indptr[gv + 1]):
            if in_subset[indices[ei]]:
                cnt += 1
        sub_indptr[li + 1] = cnt
    for li in range(n_local):
        sub_indptr[li + 1] += sub_indptr[li]
    n_edges = sub_indptr[n_local]
    sub_indices = np.empty(n_edges, dtype=np.int32)
    sub_data = np.empty(n_edges, dtype=np.float32)
    for li in range(n_local):
        gv = vertex_subset[li]
        out = sub_indptr[li]
        for ei in range(indptr[gv], indptr[gv + 1]):
            gu = indices[ei]
            if in_subset[gu]:
                sub_indices[out] = vertex_relabel[gu]
                sub_data[out] = data[ei]
                out += 1
    return sub_indptr, sub_indices, sub_data


def metis_dyadic_partition(
    affinity_graph: csr_matrix,
    *,
    max_depth: int,
    min_cell_size: int = 2,
    weight_scale: float = 1e6,
) -> Hierarchy:
    """Recursive METIS-2 bisection of `affinity_graph` into a dyadic Hierarchy.

    Parameters
    ----------
    affinity_graph : symmetric (n, n) CSR. Use `build_knn_affinity_graph`.
    max_depth : hard cap on recursion depth (root is depth 0).
    min_cell_size : a cell is not split if either child would have < this many
        points. Default 2 is the absolute floor METIS can handle; raise it if
        you want fatter leaves without relying on `max_depth` alone.
    weight_scale : float affinities are multiplied by this and rounded to
        positive integers for METIS edge weights.
    """
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0, got {max_depth}")
    if min_cell_size < 2:
        raise ValueError(
            f"min_cell_size must be >= 2 (METIS floor), got {min_cell_size}"
        )

    n = affinity_graph.shape[0]
    indptr = np.ascontiguousarray(affinity_graph.indptr, dtype=np.int32)
    indices = np.ascontiguousarray(affinity_graph.indices, dtype=np.int32)
    data = np.ascontiguousarray(affinity_graph.data, dtype=np.float32)

    # Reusable scratch buffers (allocated once, reused across recursive calls).
    in_subset = np.zeros(n, dtype=np.bool_)
    vertex_relabel = np.empty(n, dtype=np.int32)

    depth_counter: dict[int, int] = defaultdict(int)

    def make_leaf(subset: np.ndarray, node_id: tuple[int, int]) -> HierarchyNode:
        return HierarchyNode(
            idxs=subset.astype(np.int64, copy=False),
            node_id=node_id,
            children=[],
        )

    def build(subset: np.ndarray, depth: int) -> HierarchyNode:
        node_id = (depth, depth_counter[depth])
        depth_counter[depth] += 1
        n_local = subset.shape[0]

        if depth >= max_depth or n_local < 2 * min_cell_size:
            return make_leaf(subset, node_id)

        # Set mask + relabel for this subset, extract induced subgraph, clear.
        in_subset[subset] = True
        vertex_relabel[subset] = np.arange(n_local, dtype=np.int32)
        sub_indptr, sub_indices, sub_data = _extract_induced_subgraph(
            indptr,
            indices,
            data,
            in_subset,
            vertex_relabel,
            subset,
        )
        in_subset[subset] = False

        # No edges to cut → can't split; treat as leaf.
        if sub_indices.size == 0:
            return make_leaf(subset, node_id)

        # METIS edge weights must be positive integers.
        eweights_f = sub_data * weight_scale
        eweights = np.maximum(np.rint(eweights_f), 1.0).astype(np.int64)
        adj = pymetis.CSRAdjacency(adj_starts=sub_indptr, adjacent=sub_indices)
        result = pymetis.part_graph(2, adjacency=adj, eweights=eweights)
        membership = np.asarray(result.vertex_part, dtype=np.int8)
        left_local = np.flatnonzero(membership == 0).astype(np.int32, copy=False)
        right_local = np.flatnonzero(membership == 1).astype(np.int32, copy=False)

        if left_local.size < min_cell_size or right_local.size < min_cell_size:
            return make_leaf(subset, node_id)

        left = build(subset[left_local], depth + 1)
        right = build(subset[right_local], depth + 1)
        node = HierarchyNode(
            idxs=subset.astype(np.int64, copy=False),
            node_id=node_id,
            children=[left, right],
        )
        left.parent = node
        right.parent = node
        return node

    logger.info(
        "metis_dyadic_partition: n=%d max_depth=%d min_cell_size=%d nnz=%d",
        n,
        max_depth,
        min_cell_size,
        affinity_graph.nnz,
    )
    root = build(np.arange(n, dtype=np.int32), 0)

    nodes_by_id: dict[tuple[int, int], HierarchyNode] = {}
    leaves: list[HierarchyNode] = []
    height_holder = [0]

    def visit(node: HierarchyNode, depth: int) -> None:
        nodes_by_id[node.node_id] = node
        height_holder[0] = max(height_holder[0], depth + 1)
        if not node.children:
            leaves.append(node)
        for c in node.children:
            visit(c, depth + 1)

    visit(root, 0)
    logger.info(
        "metis_dyadic_partition: built tree with %d nodes, %d leaves, height=%d",
        len(nodes_by_id),
        len(leaves),
        height_holder[0],
    )

    return Hierarchy(
        root=root,
        n_samples=n,
        height=height_holder[0],
        nodes_by_id=nodes_by_id,
        leaves=leaves,
    )


# ---------- 3. Convenience wrapper ----------


def build_dyadic_hierarchy(
    X: np.ndarray,
    *,
    max_depth: int,
    k: int = 50,
    min_cell_size: int = 2,
    weight_scale: float = 1e6,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Hierarchy:
    """End-to-end: kNN affinity graph from X, then dyadic METIS partition.

    Pre-build the affinity graph yourself with `build_knn_affinity_graph` if
    you want to cache it (`scipy.sparse.save_npz`) for repeated experiments
    on the same X.
    """
    graph = build_knn_affinity_graph(
        X,
        k=k,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return metis_dyadic_partition(
        graph,
        max_depth=max_depth,
        min_cell_size=min_cell_size,
        weight_scale=weight_scale,
    )
