"""Adapters that turn hierarchical-clustering outputs into a Hierarchy.

A `Hierarchy` is a pure tree-structure object: nodes carry only point indices
and parent/child pointers. GMRA owns all model state separately.

Current adapters:
- `evoc_to_hierarchy(clusterer)`: wraps a fitted `evoc.EVoC`.

Future: `fast_hdbscan_to_hierarchy(clusterer)` — needs per-level cuts derived
from the condensed tree's lambda values; defer until needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HierarchyNode:
    idxs: np.ndarray
    node_id: tuple[int, int]
    children: list["HierarchyNode"] = field(default_factory=list)
    parent: "HierarchyNode | None" = None


@dataclass
class Hierarchy:
    root: HierarchyNode
    n_samples: int
    height: int
    nodes_by_id: dict[tuple[int, int], HierarchyNode]
    leaves: list[HierarchyNode]


def evoc_to_hierarchy(clusterer) -> Hierarchy:
    """Build a Hierarchy from a fitted `evoc.EVoC` clusterer.

    EVoC conventions used:
    - `cluster_layers_[0]` is finest (leaves); the last layer is coarsest.
    - `cluster_tree_` keys are parents, values are child lists. Node IDs in
      EVoC are `(layer_idx, cluster_id)`.
    - A synthetic EVoC root sits at `(len(cluster_layers_), 0)` collecting
      orphan top-level clusters. We propagate it as our root if present;
      otherwise we synthesize one above the discovered tree roots.
    - `node_id` is left as the EVoC `(layer_idx, cluster_id)` tuple — globally
      unique across the tree.
    - Noise points (`-1`) are excluded from every node's `idxs`. The
      nearest-leaf router at transform time still assigns them to a leaf.
    """
    tree = clusterer.cluster_tree_
    cluster_layers = clusterer.cluster_layers_
    n_layers = len(cluster_layers)
    n_samples = cluster_layers[0].shape[0]

    # Find tree roots: nodes that appear as parents but never as children.
    parents_in_tree: set[tuple[int, int]] = set(tree.keys())
    children_in_tree: set[tuple[int, int]] = set()
    for kids in tree.values():
        children_in_tree.update(kids)
    discovered_roots = [n for n in parents_in_tree if n not in children_in_tree]
    if not discovered_roots:
        raise ValueError(
            "EVoC cluster_tree_ has no parent-only nodes; cannot find a root"
        )

    def build(evoc_id: tuple[int, int]) -> HierarchyNode:
        layer, cluster = evoc_id
        children_evoc_ids = list(tree.get(evoc_id, []))
        children = [build(cid) for cid in children_evoc_ids]
        if 0 <= layer < n_layers:
            idxs = np.flatnonzero(cluster_layers[layer] == cluster)
        elif children:
            idxs = np.unique(np.concatenate([c.idxs for c in children]))
        else:
            idxs = np.zeros(0, dtype=np.int64)
        node = HierarchyNode(
            idxs=idxs,
            node_id=(int(layer), int(cluster)),
            children=children,
        )
        for c in children:
            c.parent = node
        return node

    if len(discovered_roots) == 1:
        root = build(discovered_roots[0])
    else:
        sub_roots = [build(rid) for rid in discovered_roots]
        all_idxs = np.unique(np.concatenate([s.idxs for s in sub_roots]))
        root = HierarchyNode(
            idxs=all_idxs,
            node_id=(n_layers + 1, 0),
            children=sub_roots,
        )
        for s in sub_roots:
            s.parent = root

    leaves: list[HierarchyNode] = []
    nodes_by_id: dict[tuple[int, int], HierarchyNode] = {}
    height_holder = [0]

    def visit(node: HierarchyNode, depth: int) -> None:
        nodes_by_id[node.node_id] = node
        height_holder[0] = max(height_holder[0], depth + 1)
        if not node.children:
            leaves.append(node)
        for c in node.children:
            visit(c, depth + 1)

    visit(root, 0)

    return Hierarchy(
        root=root,
        n_samples=n_samples,
        height=height_holder[0],
        nodes_by_id=nodes_by_id,
        leaves=leaves,
    )


def build_hierarchy(
    tree_type: str,
    X: np.ndarray,
    *,
    evoc_kwargs: dict | None = None,
    dyadic_max_depth: int = 12,
    dyadic_k: int = 50,
    dyadic_min_cell_size: int = 25,
) -> tuple[Hierarchy, object | None]:
    """Dispatch hierarchy construction by ``tree_type``.

    Fits the underlying clusterer (EVoC) or builds the dyadic tree internally.
    Callers don't need to do any clustering work themselves — passing
    ``tree_type`` decides everything downstream.

    Parameters
    ----------
    tree_type : ``"evoc"`` or ``"dyadic"``.
    X : ambient embeddings.
    evoc_kwargs : kwargs forwarded to ``evoc.EVoC``; used only for
        ``"evoc"``. Defaults to an empty dict.
    dyadic_max_depth, dyadic_k, dyadic_min_cell_size : dyadic params, used
        only for ``"dyadic"``.

    Returns
    -------
    hierarchy : Hierarchy
    clusterer : fitted ``evoc.EVoC`` when ``tree_type == 'evoc'``, else
        ``None``. Exposed so callers can run EVoC-specific diagnostics
        without refitting; GMRA itself doesn't read it.
    """
    if tree_type == "evoc":
        import evoc

        clusterer = evoc.EVoC(**(evoc_kwargs or {}))
        clusterer.fit_predict(X)
        return evoc_to_hierarchy(clusterer), clusterer
    if tree_type == "dyadic":
        # Lazy import: dyadic_decomposition pulls in numba/pymetis/pynndescent.
        from dyadic_decomposition import build_dyadic_hierarchy

        hierarchy = build_dyadic_hierarchy(
            X,
            max_depth=dyadic_max_depth,
            k=dyadic_k,
            min_cell_size=dyadic_min_cell_size,
        )
        return hierarchy, None
    raise ValueError(f"unknown tree_type: {tree_type!r}")
