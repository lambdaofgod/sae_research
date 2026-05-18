"""GMRA (geometric multiresolution analysis) over a hierarchical-clustering tree.

This module is a port of the wavelet machinery from
``Geometric-Multi-Resolution-Analysis/src/dyadictree.py`` decoupled from the
reference's cover-tree dependency. The tree comes in as a ``Hierarchy``
(see ``tree_adapters.py``); GMRA computes per-node local PCA + wavelets and
exposes a leaf-routed sparse encoder/decoder.

The chosen encoder produces, for each x:
    1. A leaf assignment L (nearest leaf center).
    2. Sparse codes in R^{n_atoms} with non-zeros only on atoms belonging to
       nodes on path(L).
This is the natural multiscale wavelet decomposition of x against its leaf's
root-to-leaf chain, scattered into a fixed-width sparse vector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.utils.extmath import randomized_svd
from tqdm.auto import tqdm

from tree_adapters import Hierarchy, HierarchyNode

logger = logging.getLogger(__name__)


def path_to_root(node: HierarchyNode) -> list[HierarchyNode]:
    """Root-to-`node` chain. Ported from utils.path."""
    chain = [node]
    cur = node.parent
    while cur is not None:
        chain.append(cur)
        cur = cur.parent
    chain.reverse()
    return chain


class BasisDimStrategy(Protocol):
    """Decide how many local-PCA components to keep at a node.

    The strategy sees the singular-value spectrum from the local SVD plus a
    ``trailing_sigma`` representing the energy in dimensions beyond ``max_dim``
    (the residual Frobenius energy of X_centered not captured by the truncated
    SVD). This is enough to express either fixed-k or energy-threshold rules.
    """

    def __call__(
        self,
        sigmas: np.ndarray,
        trailing_sigma: float,
        n_points: int,
        ambient_dim: int,
        is_leaf: bool,
    ) -> int: ...


@dataclass
class FixedManifoldDim:
    """Truncate every node's basis to ``k`` (clipped to numerical rank)."""

    k: int

    def __call__(self, sigmas, trailing_sigma, n_points, ambient_dim, is_leaf):
        return min(self.k, len(sigmas), int(np.sum(sigmas > 0)))


@dataclass
class EnergyThreshold:
    """Keep enough components for relative energy >= 1 - eps.

    Mirrors the ``mindim`` selector in helpers.py:4-20 (errortype="relative").
    The reference appends a synthetic "trailing" singular value capturing
    energy beyond the truncated SVD; we do the same so the threshold sees
    the full Frobenius norm.
    """

    eps: float

    def __call__(self, sigmas, trailing_sigma, n_points, ambient_dim, is_leaf):
        augmented = np.concatenate([sigmas, [trailing_sigma]])
        total = float(np.sum(augmented**2))
        if total <= 0:
            return 0
        tol = self.eps * total
        remaining = total
        dim = 0
        while dim < len(sigmas) and remaining > tol:
            remaining -= augmented[dim] ** 2
            dim += 1
        return dim


def local_pca(
    X_centered: np.ndarray,
    max_dim: int,
    dim_strategy: BasisDimStrategy,
    is_leaf: bool,
    random_state: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Truncated PCA via randomized SVD; strategy picks the final dim.

    Uses sklearn's ``randomized_svd`` so the left singular matrix is capped
    at ``(n, max_dim)`` instead of the ``(n, min(n,d))`` that full SVD would
    allocate — important for large root clusters (e.g. n ≈ 1e6).

    Trailing energy beyond ``max_dim`` is estimated as
    ``‖X_centered‖_F² - Σ σ_i²`` and passed to the strategy so the
    energy-threshold rule still works.

    Parameters
    ----------
    X_centered : (n_points, d)
        Points already centered on their mean.
    max_dim : int
        Hard ceiling on SVD rank.

    Returns
    -------
    basis : (k, d)
        Top-k right singular vectors. ``k`` is the strategy's choice.
    sigmas : (k,)
        Singular values for the kept components (unnormalized — raw SVD σ).
    """
    n, d = X_centered.shape
    full_k = min(n, d, max_dim)
    if full_k == 0:
        return np.zeros((0, d)), np.zeros((0,))

    if full_k >= min(n, d):
        # Truncated == full; use exact SVD, cheaper than randomized at this size.
        _, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        s = s[:full_k]
        Vt = Vt[:full_k]
    else:
        _, s, Vt = randomized_svd(
            X_centered, n_components=full_k, random_state=random_state
        )

    # Frobenius energy without allocating an (n, d) temp.
    total_energy = float(np.einsum("ij,ij->", X_centered, X_centered))
    trailing_energy = max(total_energy - float(np.sum(s**2)), 0.0)
    size_norm = np.sqrt(max(n, 1))
    sigmas_normalized = s / size_norm
    trailing_sigma = np.sqrt(trailing_energy) / size_norm

    requested = dim_strategy(
        sigmas=sigmas_normalized,
        trailing_sigma=trailing_sigma,
        n_points=n,
        ambient_dim=d,
        is_leaf=is_leaf,
    )
    dim = min(requested, len(s), int(np.sum(s > 0)))
    logger.debug(
        "local_pca: n=%d d=%d max_dim=%d is_leaf=%s -> kept_dim=%d",
        n,
        d,
        max_dim,
        is_leaf,
        dim,
    )
    return Vt[:dim], s[:dim]


class GMRA:
    """Multiscale wavelet model over a fixed Hierarchy.

    After ``fit``, exposes:
      - ``transform(X, adaptive=True, criterion='center') -> (codes, paths)``:
        sparse encoder. With ``adaptive=True`` (default), each sample descends
        greedily and stops where no child reduces the routing residual.
        With ``adaptive=False``, samples are routed via nearest-leaf Voronoi.
      - ``inverse_transform(codes, paths) -> X_hat``: sparse decoder.
      - ``wavelets``: (n_atoms, d) stack of scaling + wavelet basis rows. Not a
        dictionary in the SAE sense — atoms are tied to specific tree nodes
        and only usable when a sample is routed through that node.
      - ``leaf_base[L_id]``: per-leaf affine offset (kept for compatibility;
        adaptive decoder computes path-specific bases on the fly).
    """

    def __init__(
        self,
        hierarchy: Hierarchy,
        basis_dim_strategy: BasisDimStrategy,
        max_dim: int,
        threshold: float = 0.5,
    ):
        self.hierarchy = hierarchy
        self.basis_dim_strategy = basis_dim_strategy
        self.max_dim = max_dim
        self.threshold = threshold

        # per-node model state (keyed by node_id) — populated by fit:
        self.center: dict[tuple[int, int], np.ndarray] = {}
        self.basis: dict[tuple[int, int], np.ndarray] = {}
        self.wav_basis: dict[tuple[int, int], np.ndarray] = {}
        self.wav_consts: dict[tuple[int, int], np.ndarray] = {}
        self.sigmas: dict[tuple[int, int], np.ndarray] = {}

        # wavelet stack — populated after make_transform completes:
        self.atom_slice: dict[tuple[int, int], slice] = {}
        self.wavelets: np.ndarray | None = None  # (n_atoms, d)
        self.leaf_base: dict[tuple[int, int], np.ndarray] = {}
        self.leaf_centers: np.ndarray | None = None  # (n_leaves, d)
        self.leaf_ids: list[tuple[int, int]] = []
        self.n_atoms: int = 0
        self._fitted = False

    # ----- fit -----

    def fit(self, X: np.ndarray) -> "GMRA":
        logger.info(
            "GMRA.fit: X.shape=%s dtype=%s, hierarchy: %d nodes, %d leaves, height=%d",
            X.shape,
            X.dtype,
            len(self.hierarchy.nodes_by_id),
            len(self.hierarchy.leaves),
            self.hierarchy.height,
        )
        self._make_basis(X)
        self._make_transform()
        self._build_wavelets()
        self._fitted = True
        logger.info(
            "GMRA.fit done: n_atoms=%d, wavelets=%s",
            self.n_atoms,
            None if self.wavelets is None else self.wavelets.shape,
        )
        return self

    def _make_basis(self, X: np.ndarray) -> None:
        """Top-down recursion: at each node, run local PCA on X[node.idxs]."""
        nodes = self._iter_nodes_topdown()
        for node in tqdm(
            nodes, desc="make_basis", unit="node", mininterval=5, miniters=1
        ):
            idxs = node.idxs
            if idxs.size == 0:
                # Empty node — degenerate. Store zero basis at origin.
                d = X.shape[1]
                self.center[node.node_id] = np.zeros(d)
                self.basis[node.node_id] = np.zeros((0, d))
                self.sigmas[node.node_id] = np.zeros((0,))
                continue
            X_sub = X[idxs]
            center = X_sub.mean(axis=0)
            X_centered = X_sub - center
            is_leaf = len(node.children) == 0
            basis, sigmas = local_pca(
                X_centered,
                max_dim=self.max_dim,
                dim_strategy=self.basis_dim_strategy,
                is_leaf=is_leaf,
            )
            self.center[node.node_id] = center
            self.basis[node.node_id] = basis
            self.sigmas[node.node_id] = sigmas

    def _make_transform(self) -> None:
        """For every parent->child edge, compute child's wavelet basis and constant.

        wav_basis[child] = orthonormal basis for the rows of
        (Phi_child - Phi_child @ Phi_parent.T @ Phi_parent), truncated to
        components with singular value > self.threshold.
        wav_consts[child] = (c_child - c_parent) projected onto orthogonal
        complement of Phi_parent.
        Root gets zero-row wav_basis and zero wav_consts (it has no parent).
        """
        d = self._ambient_dim()
        root_id = self.hierarchy.root.node_id
        self.wav_basis[root_id] = np.zeros((0, d))
        self.wav_consts[root_id] = np.zeros(d)

        nodes = self._iter_nodes_topdown()
        n_edges = sum(len(node.children) for node in nodes)
        with tqdm(
            total=n_edges, desc="make_transform", unit="edge", mininterval=5
        ) as pbar:
            for node in nodes:
                for child in node.children:
                    self._compute_child_wavelet(parent=node, child=child)
                    pbar.update(1)

    def _compute_child_wavelet(
        self, parent: HierarchyNode, child: HierarchyNode
    ) -> None:
        d = self._ambient_dim()
        Phi_parent = self.basis[parent.node_id]  # (k_p, d)
        Phi_child = self.basis[child.node_id]  # (k_c, d)
        c_parent = self.center[parent.node_id]  # (d,)
        c_child = self.center[child.node_id]  # (d,)

        if Phi_child.size == 0:
            self.wav_basis[child.node_id] = np.zeros((0, d))
            self.wav_consts[child.node_id] = np.zeros(d)
            return

        # Y rows = child basis vectors with parent-subspace component removed.
        if Phi_parent.size == 0:
            Y = Phi_child
        else:
            Y = Phi_child - Phi_child @ Phi_parent.T @ Phi_parent

        # SVD on Y.T (shape (d, k_c)). U holds left singular vectors in ambient dim.
        if Y.size == 0 or np.allclose(Y, 0):
            self.wav_basis[child.node_id] = np.zeros((0, d))
        else:
            U, s, _ = np.linalg.svd(Y.T, full_matrices=False)
            kept = int(np.sum(s > self.threshold))
            if kept > 0:
                self.wav_basis[child.node_id] = U[:, :kept].T  # (kept, d)
            else:
                self.wav_basis[child.node_id] = np.zeros((0, d))

        # wav_consts: shift c_child - c_parent projected onto orth complement of parent.
        t = c_child - c_parent
        if Phi_parent.size == 0:
            self.wav_consts[child.node_id] = t
        else:
            self.wav_consts[child.node_id] = t - Phi_parent.T @ (Phi_parent @ t)

    def _build_wavelets(self) -> None:
        """Assign atom slices, stack the wavelets matrix, precompute per-leaf bases."""
        root_id = self.hierarchy.root.node_id

        # Atom layout: root scaling atoms first, then wavelet atoms for every other node
        # in top-down enumeration order. Order is stable across fits.
        wavelet_blocks: list[np.ndarray] = []
        cursor = 0
        root_basis = self.basis[root_id]
        self.atom_slice[root_id] = slice(cursor, cursor + root_basis.shape[0])
        cursor += root_basis.shape[0]
        wavelet_blocks.append(root_basis)

        for node in self._iter_nodes_topdown():
            if node.node_id == root_id:
                continue
            wav = self.wav_basis[node.node_id]
            self.atom_slice[node.node_id] = slice(cursor, cursor + wav.shape[0])
            cursor += wav.shape[0]
            wavelet_blocks.append(wav)

        self.n_atoms = cursor
        if wavelet_blocks:
            self.wavelets = np.vstack(wavelet_blocks)
        else:
            self.wavelets = np.zeros((0, self._ambient_dim()))

        # Per-leaf bases + leaf-center matrix for nearest-leaf routing.
        leaves = self.hierarchy.leaves
        self.leaf_ids = [leaf.node_id for leaf in leaves]
        self.leaf_centers = np.stack([self.center[leaf.node_id] for leaf in leaves])
        c_root = self.center[root_id]
        for leaf in leaves:
            base = c_root.copy()
            for ancestor in path_to_root(leaf):
                if ancestor.node_id == root_id:
                    continue
                base = base + self.wav_consts[ancestor.node_id]
            self.leaf_base[leaf.node_id] = base

    # ----- transform / inverse -----

    def transform(
        self,
        X: np.ndarray,
        adaptive: bool = True,
        criterion: str = "center",
    ) -> tuple[csr_matrix, list[list[tuple[int, int]]]]:
        """Encode X to sparse codes plus per-sample paths.

        Parameters
        ----------
        adaptive : bool
            If True (default), each sample descends the tree greedily and
            stops at the first node where no child reduces the routing
            residual. If False, samples are routed via nearest-leaf-center
            Voronoi and walk the full root→leaf path.
        criterion : {"center", "projection"}
            How to measure residual for the routing decision (adaptive only).
            ``"center"`` uses ``‖x − c_N‖²``; ``"projection"`` uses
            ``‖(I − Φ_Nᵀ Φ_N)(x − c_N)‖²``.

        Returns
        -------
        codes : csr_matrix, shape (n_samples, n_atoms)
            Non-zero only at atom slices for nodes on each sample's path.
        paths : list of lists of node_id
            ``paths[i]`` = root → stop-node chain for sample i.
        """
        if not self._fitted:
            raise RuntimeError("GMRA.transform called before fit")
        if criterion not in ("center", "projection"):
            raise ValueError(
                f"criterion must be 'center' or 'projection', got {criterion!r}"
            )

        if adaptive:
            stop_node_ids = self._adaptive_route(X, criterion)
        else:
            leaf_positions = cdist(X, self.leaf_centers).argmin(axis=1)
            stop_node_ids = [self.leaf_ids[la] for la in leaf_positions]

        nodes_by_id = self.hierarchy.nodes_by_id
        paths = [
            [n.node_id for n in path_to_root(nodes_by_id[sid])] for sid in stop_node_ids
        ]

        codes = self._encode_from_stop_nodes(X, stop_node_ids)
        return codes, paths

    def _adaptive_route(self, X: np.ndarray, criterion: str) -> list[tuple[int, int]]:
        """Per-sample greedy descent. Returns the stop-node id for each sample.

        At each node, evaluate the routing residual at that node and all its
        children. If any child has strictly smaller residual, descend to the
        best one. Else stop here.

        Vectorized: child-residual computation uses ``cdist`` (center) or
        Pythagorean (projection) to keep intermediates small; bucket
        assignment for the next level is fully numpy, not a Python sample loop.
        """
        n = X.shape[0]
        # stop[i] stores int index into self._adaptive_node_list (built lazily)
        if not hasattr(self, "_adaptive_node_list"):
            self._adaptive_node_list = list(self.hierarchy.nodes_by_id.keys())
            self._adaptive_node_to_idx = {
                nid: i for i, nid in enumerate(self._adaptive_node_list)
            }

        stop_idx = np.full(n, -1, dtype=np.int64)
        bucket: dict[tuple[int, int], np.ndarray] = {
            self.hierarchy.root.node_id: np.arange(n, dtype=np.int64)
        }
        progress = tqdm(desc=f"adaptive-route-{criterion}", unit="hop", mininterval=5)

        while bucket:
            progress.update(1)
            next_bucket: dict[tuple[int, int], np.ndarray] = {}
            for node_id, sample_idxs in bucket.items():
                node = self.hierarchy.nodes_by_id[node_id]
                B = sample_idxs.size
                if not node.children:
                    stop_idx[sample_idxs] = self._adaptive_node_to_idx[node_id]
                    continue
                X_batch = X[sample_idxs]
                curr_res = self._node_residual(X_batch, node_id, criterion)
                child_ids = [c.node_id for c in node.children]
                child_res = self._children_residuals(X_batch, child_ids, criterion)
                best_idx = child_res.argmin(axis=1)
                best_res = child_res[np.arange(B), best_idx]
                descend = best_res < curr_res

                # Vectorized: samples that stop here
                stay_samples = sample_idxs[~descend]
                if stay_samples.size > 0:
                    stop_idx[stay_samples] = self._adaptive_node_to_idx[node_id]

                # Vectorized: bucket descending samples by chosen child
                if descend.any():
                    descend_samples = sample_idxs[descend]
                    descend_targets = best_idx[descend]
                    # Group by target child index
                    order = np.argsort(descend_targets, kind="stable")
                    sorted_samples = descend_samples[order]
                    sorted_targets = descend_targets[order]
                    splits = np.flatnonzero(np.diff(sorted_targets)) + 1
                    groups = np.split(sorted_samples, splits)
                    unique_targets = np.concatenate(
                        ([sorted_targets[0]], sorted_targets[splits])
                    )
                    for grp_samples, child_pos in zip(groups, unique_targets):
                        next_bucket[child_ids[int(child_pos)]] = grp_samples

            bucket = next_bucket
        progress.close()

        # Convert int-index stops back to node_id tuples
        return [self._adaptive_node_list[int(i)] for i in stop_idx]

    def _node_residual(
        self,
        X_batch: np.ndarray,
        node_id: tuple[int, int],
        criterion: str,
    ) -> np.ndarray:
        """Squared residual for X_batch at one node.

        For ``criterion='projection'`` uses the Pythagorean identity
        ``‖x−c‖² − ‖(x−c) Φᵀ‖²`` so the intermediate is ``(B, k)`` not ``(B, d)``.
        """
        c = self.center[node_id]
        diffs = X_batch - c  # (B, d)
        norm_sq = np.einsum("ij,ij->i", diffs, diffs)  # (B,)
        if criterion == "center":
            return norm_sq
        Phi = self.basis[node_id]
        if Phi.shape[0] == 0:
            return norm_sq
        proj = diffs @ Phi.T  # (B, k) — small
        proj_norm_sq = np.einsum("ij,ij->i", proj, proj)
        # Numerical guard: floating error can leave a tiny negative.
        return np.maximum(norm_sq - proj_norm_sq, 0.0)

    def _children_residuals(
        self,
        X_batch: np.ndarray,
        child_ids: list[tuple[int, int]],
        criterion: str,
    ) -> np.ndarray:
        """Squared residuals for X_batch against every child of one node.

        Returns ``(B, n_children)``. Both criteria use a single batched
        path: center uses ``cdist``; projection stacks all children's
        bases and does one big ``X @ Phi_all.T`` per sample-chunk, then
        column-slices to recover per-child projection magnitudes.
        """
        centers = np.stack([self.center[cid] for cid in child_ids])  # (C, d)
        norm_sq_all = cdist(X_batch, centers, metric="sqeuclidean")  # (B, C)
        if criterion == "center":
            return norm_sq_all
        return self._children_projection_residuals(
            X_batch=X_batch,
            child_ids=child_ids,
            centers=centers,
            norm_sq_all=norm_sq_all,
        )

    def _children_projection_residuals(
        self,
        X_batch: np.ndarray,
        child_ids: list[tuple[int, int]],
        centers: np.ndarray,
        norm_sq_all: np.ndarray,
        chunk_cells: int = 64_000_000,
    ) -> np.ndarray:
        """Projection-criterion residuals for all children of one node in
        one batched matmul.

        Uses ``‖x − c‖² − ‖(x − c) Φᵀ‖²`` with the Pythagorean identity,
        and computes ``X @ Φ_all.T`` once with all children's bases stacked
        row-wise into ``Φ_all`` of shape ``(Σ k_c, d)``. The chunked sample
        loop bounds the intermediate to ``chunk × Σ k_c`` cells.

        Parameters
        ----------
        chunk_cells : int
            Memory budget for the ``(chunk, Σ k_c)`` projection intermediate.
            Default ~64M cells ≈ 256 MB at float32 / 512 MB at float64.
        """
        B = X_batch.shape[0]
        C = len(child_ids)

        bases = [self.basis[cid] for cid in child_ids]
        nz_indices = np.array([j for j, b in enumerate(bases) if b.shape[0] > 0])
        if nz_indices.size == 0:
            # No child has any basis directions — projection adds nothing.
            return norm_sq_all

        # Stack non-empty child bases row-wise into Phi_all (Σ k, d).
        nz_bases = [bases[int(j)] for j in nz_indices]
        Phi_all = np.vstack(nz_bases)
        sum_k = Phi_all.shape[0]
        col_offsets = np.concatenate(([0], np.cumsum([b.shape[0] for b in nz_bases])))

        # Per non-empty child: c @ Φ_c.T (small vector, used to recenter
        # the projection columns). Computed once.
        center_projs = [centers[int(j)] @ bases[int(j)].T for j in nz_indices]

        proj_norm_sq = np.zeros((B, C), dtype=np.float64)
        chunk = max(1, chunk_cells // max(sum_k, 1))
        for start in range(0, B, chunk):
            end = min(start + chunk, B)
            # One BLAS matmul covers every child at once for this sample chunk.
            X_proj_all = X_batch[start:end] @ Phi_all.T  # (chunk, Σ k)
            for slot, child_pos in enumerate(nz_indices):
                s = col_offsets[slot]
                e = col_offsets[slot + 1]
                diff = X_proj_all[:, s:e] - center_projs[slot]  # (chunk, k_c)
                proj_norm_sq[start:end, int(child_pos)] = np.einsum(
                    "ij,ij->i", diff, diff
                )

        # Numerical guard: float error can leave a tiny negative.
        return np.maximum(norm_sq_all - proj_norm_sq, 0.0)

    def _encode_from_stop_nodes(
        self,
        X: np.ndarray,
        stop_node_ids: list[tuple[int, int]],
    ) -> csr_matrix:
        """For each sample, compute wavelet/scaling coefficients along its
        root → stop-node path. The stop node acts as the "leaf" for the
        single-sample fgwt math."""
        from collections import defaultdict

        n = X.shape[0]
        samples_by_stop: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, sid in enumerate(stop_node_ids):
            samples_by_stop[sid].append(i)

        rows_acc: list[np.ndarray] = []
        cols_acc: list[np.ndarray] = []
        data_acc: list[np.ndarray] = []
        nodes_by_id = self.hierarchy.nodes_by_id
        root_id = self.hierarchy.root.node_id

        iterator = samples_by_stop.items()
        if len(samples_by_stop) > 16:
            iterator = tqdm(
                samples_by_stop.items(),
                desc="encode-stop-nodes",
                unit="group",
                total=len(samples_by_stop),
            )

        for stop_id, sample_list in iterator:
            sample_idxs = np.asarray(sample_list)
            stop_node = nodes_by_id[stop_id]
            chain = path_to_root(stop_node)  # root → stop
            Phi_S = self.basis[stop_id]
            c_S = self.center[stop_id]
            X_batch = X[sample_idxs]
            if Phi_S.shape[0] > 0:
                pJx = (X_batch - c_S) @ Phi_S.T
            else:
                pJx = np.zeros((len(sample_idxs), 0), dtype=X.dtype)

            # Wavelet block at stop node, if it has one and is not root
            if stop_id != root_id and Phi_S.shape[0] > 0:
                self._scatter_wavelet_block(
                    node_id=stop_id,
                    pjx_local=pJx,
                    Phi_node=Phi_S,
                    sample_idxs=sample_idxs,
                    rows_acc=rows_acc,
                    cols_acc=cols_acc,
                    data_acc=data_acc,
                )

            # Ancestors (excluding the stop node itself)
            for ancestor in reversed(chain[:-1]):
                anc_id = ancestor.node_id
                Phi_n = self.basis[anc_id]
                c_n = self.center[anc_id]
                if Phi_n.shape[0] == 0:
                    continue
                if Phi_S.shape[0] > 0:
                    pjx_n = pJx @ Phi_S @ Phi_n.T + (c_S - c_n) @ Phi_n.T
                else:
                    pjx_n = (X_batch - c_n) @ Phi_n.T
                if anc_id == root_id:
                    self._scatter_block(
                        block=pjx_n,
                        slc=self.atom_slice[root_id],
                        sample_idxs=sample_idxs,
                        rows_acc=rows_acc,
                        cols_acc=cols_acc,
                        data_acc=data_acc,
                    )
                else:
                    self._scatter_wavelet_block(
                        node_id=anc_id,
                        pjx_local=pjx_n,
                        Phi_node=Phi_n,
                        sample_idxs=sample_idxs,
                        rows_acc=rows_acc,
                        cols_acc=cols_acc,
                        data_acc=data_acc,
                    )

        if rows_acc:
            rows = np.concatenate(rows_acc)
            cols = np.concatenate(cols_acc)
            data = np.concatenate(data_acc)
        else:
            rows = np.zeros(0, dtype=np.int64)
            cols = np.zeros(0, dtype=np.int64)
            data = np.zeros(0, dtype=np.float64)
        return csr_matrix((data, (rows, cols)), shape=(n, self.n_atoms))

    def _scatter_wavelet_block(
        self,
        node_id: tuple[int, int],
        pjx_local: np.ndarray,
        Phi_node: np.ndarray,
        sample_idxs: np.ndarray,
        rows_acc: list[np.ndarray],
        cols_acc: list[np.ndarray],
        data_acc: list[np.ndarray],
    ) -> None:
        Psi = self.wav_basis[node_id]
        if Psi.shape[0] == 0:
            return
        q = pjx_local @ Phi_node @ Psi.T  # (m, m_node)
        self._scatter_block(
            block=q,
            slc=self.atom_slice[node_id],
            sample_idxs=sample_idxs,
            rows_acc=rows_acc,
            cols_acc=cols_acc,
            data_acc=data_acc,
        )

    @staticmethod
    def _scatter_block(
        block: np.ndarray,
        slc: slice,
        sample_idxs: np.ndarray,
        rows_acc: list[np.ndarray],
        cols_acc: list[np.ndarray],
        data_acc: list[np.ndarray],
    ) -> None:
        m, dim = block.shape
        if dim == 0 or m == 0:
            return
        rows = np.repeat(sample_idxs, dim)
        cols = np.tile(np.arange(slc.start, slc.stop), m)
        rows_acc.append(rows)
        cols_acc.append(cols)
        data_acc.append(block.ravel())

    def inverse_transform(
        self,
        codes: csr_matrix,
        paths: list[list[tuple[int, int]]],
    ) -> np.ndarray:
        """Decode using per-sample paths produced by :meth:`transform`.

        Computes ``x̂ = codes @ wavelets + path_base`` where ``path_base`` is
        the cumulative affine offset along each sample's path:
        ``c_root + Σ wav_consts[n]`` for non-root nodes ``n`` on the path.
        """
        if not self._fitted:
            raise RuntimeError("GMRA.inverse_transform called before fit")
        n = codes.shape[0]
        d = self.wavelets.shape[1]
        root_id = self.hierarchy.root.node_id

        # Group samples by their full path (encoded as a tuple) so we compute
        # each unique path_base once. For adaptive paths there can be many
        # distinct paths; for non-adaptive the count equals the leaf count.
        from collections import defaultdict

        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, p in enumerate(paths):
            groups[tuple(p)].append(i)

        bases = np.empty((n, d), dtype=self.wavelets.dtype)
        for path_key, sample_idxs in groups.items():
            base = self.center[path_key[0]].copy()  # root's center
            for nid in path_key[1:]:
                base = base + self.wav_consts[nid]
            bases[sample_idxs] = base
        return codes @ self.wavelets + bases

    # ----- helpers -----

    def _iter_nodes_topdown(self) -> list[HierarchyNode]:
        """DFS order starting at root. Deterministic given hierarchy."""
        out: list[HierarchyNode] = []
        stack = [self.hierarchy.root]
        while stack:
            node = stack.pop()
            out.append(node)
            stack.extend(reversed(node.children))
        return out

    def _ambient_dim(self) -> int:
        # Any non-empty basis tells us d; fall back to wavelets if available.
        for basis in self.basis.values():
            if basis.size > 0:
                return basis.shape[1]
        for center in self.center.values():
            if center.size > 0:
                return center.shape[0]
        raise RuntimeError("Cannot infer ambient dim before fit")
