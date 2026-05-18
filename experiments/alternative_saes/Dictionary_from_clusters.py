# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: alternative_saes
#     language: python
#     name: alternative_saes
# ---

# %% [markdown]
# We study modification of ITDA which uses a more principled way to select dictionary atoms  that is based on kNN algorithm
#
# IDEA: instead of ad hoc selection just create the dictionary from something like kNN dendrogram
#
# This approach is inspired by GMRA (geometric multiresolution analysis) and ITDA (Inference-time decomposition of activations).
#
# ITDA proposed selecting latents from the dataset - tokens that are reconstructed poorly are added to the dictionary. GMRA works by analogy of wavelets - the space is partitioned recursively and inside each partition a local SVD provides the coefficients. The crucial part is that recursive partitions use cover trees or a similar method - such methods are a workhorse of hierarchical clustering algorithms like HDBSCAN, for which fast algorithms are available thanks to Leland McInnes and Tutte Institute.

# %%
# %%time
import evoc
import pandas as pd
import pyarrow.parquet as pq
import numpy as np

# %%
pythia_embeddings_df = pd.read_parquet("pythia70m_layer3_n10000.parquet")


# %%
# %%time
t = pq.read_table("pythia70m_layer3_n10000.parquet")

# the fixed-size-list column is just a flat float array under the hood
col = t.column("activation").combine_chunks()
pythia_embeddings_orig = col.values.to_numpy().reshape(-1, 512)

# %%
pythia_embeddings_df.head()

# %%
# %%time

# Dedup the activation matrix AND keep the metadata df aligned to it.
# np.unique returns rows in sorted order; return_index lets us reorder
# pythia_embeddings_df so row i of both objects refers to the same token.
_, unique_idxs = np.unique(pythia_embeddings_orig, axis=0, return_index=True)
pythia_embeddings = pythia_embeddings_orig[unique_idxs]
pythia_embeddings_df = (
    pythia_embeddings_df.iloc[unique_idxs]
    .drop(columns=["activation"])
    .reset_index(drop=True)
)
assert len(pythia_embeddings_df) == pythia_embeddings.shape[0]

# %%
pythia_embeddings.shape, pythia_embeddings_df.shape

# %% [markdown]
# ## Hierarchy source toggle
# Pick which method builds the GMRA hierarchy. ``build_hierarchy`` does
# ALL the clustering / tree-building work for the chosen mode — there's
# no separate EVoC fit cell. ``"evoc"`` fits EVoC internally; ``"dyadic"``
# uses METIS recursive bisection over a kNN affinity graph (paper §2.1).

# %%
TREE_TYPE = "dyadic"  # "evoc" | "dyadic"

# EVoC-only knobs (forwarded as ``evoc.EVoC(**EVOC_KWARGS)``):
EVOC_KWARGS = dict(noise_level=1.0, max_layers=10, min_samples=25)

# Dyadic-only knobs:
DYADIC_MAX_DEPTH = 12  # binary tree → up to 2**MAX_DEPTH leaves
DYADIC_K = 50  # kNN neighbors for the affinity graph (paper default)
DYADIC_MIN_CELL_SIZE = 25  # don't split if either child would be smaller

# %%
import logging
from gmra import GMRA, FixedManifoldDim
from tree_adapters import build_hierarchy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
)

# %%
print(
    f"pythia_embeddings: shape={pythia_embeddings.shape}, dtype={pythia_embeddings.dtype}, {pythia_embeddings.nbytes / 1e9:.2f} GB"
)

# Cast to float32 to halve memory at every step. Activations don't need float64.
if pythia_embeddings.dtype != np.float32:
    pythia_embeddings = pythia_embeddings.astype(np.float32)
    print(f"cast to float32: {pythia_embeddings.nbytes / 1e9:.2f} GB")

# %%
# %%time
hierarchy, clusterer = build_hierarchy(
    TREE_TYPE,
    X=pythia_embeddings,
    evoc_kwargs=EVOC_KWARGS,
    dyadic_max_depth=DYADIC_MAX_DEPTH,
    dyadic_k=DYADIC_K,
    dyadic_min_cell_size=DYADIC_MIN_CELL_SIZE,
)
print(f"[{TREE_TYPE}] n_samples covered by hierarchy: {hierarchy.n_samples}")
print(f"[{TREE_TYPE}] tree height: {hierarchy.height}")
print(f"[{TREE_TYPE}] n_leaves: {len(hierarchy.leaves)}")
print(f"[{TREE_TYPE}] total nodes: {len(hierarchy.nodes_by_id)}")

# %% [markdown]
# ## Optional: EVoC clustering diagnostics (legacy exploration)
# Runs only when ``TREE_TYPE == "evoc"`` — i.e. ``build_hierarchy`` returned
# a fitted clusterer. In dyadic mode ``clusterer is None`` and this whole
# cell no-ops cleanly. Includes the abandoned cluster-exemplar dictionary
# builds that predate the GMRA approach.

# %%
if clusterer is not None:
    import matplotlib.pyplot as plt
    from collections import Counter

    # ---- structural stats over the EVoC cluster tree ----
    print(pd.Series(clusterer.cluster_layers_[0]).value_counts())

    tree = clusterer.cluster_tree_
    all_tree_nodes = set(tree.keys())
    for kids in tree.values():
        all_tree_nodes.update(kids)
    print(f"Total nodes in tree: {len(all_tree_nodes)}")
    print(f"Tree keys (parents): {len(tree)}")
    print(f"Total edges (parent->child): {sum(len(v) for v in tree.values())}")
    nodes_per_layer = Counter(n[0] for n in all_tree_nodes)
    print(f"\nTree nodes per layer: {dict(sorted(nodes_per_layer.items()))}")
    print("\nUnique cluster IDs per layer (from cluster_layers_):")
    for i, labels in enumerate(clusterer.cluster_layers_):
        n_clusters = len(np.unique(labels[labels >= 0]))
        print(f"  Layer {i}: {n_clusters}")

    # ---- membership-strength stats ----
    print(
        "membership strength 0.9 quantile:",
        pd.Series(clusterer.membership_strength_layers_[0]).quantile(0.9),
    )
    print("membership strength shape:", clusterer.membership_strength_layers_[0].shape)

    # ---- cluster size distribution ----
    def cluster_size_distribution(clusterer, layer):
        labels = clusterer.cluster_layers_[layer]
        sizes = pd.Series(labels[labels >= 0]).value_counts().values
        fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(10, 3))
        ax_lin.hist(sizes, bins=50)
        ax_lin.set_xlabel("cluster size")
        ax_lin.set_ylabel("# clusters")
        ax_lin.set_title("linear")
        ax_log.hist(
            sizes, bins=np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 50)
        )
        ax_log.set_xscale("log")
        ax_log.set_yscale("log")
        ax_log.set_xlabel("cluster size (log)")
        ax_log.set_ylabel("# clusters (log)")
        ax_log.set_title("log-log")
        fig.suptitle(
            f"Layer {layer} — {len(sizes)} clusters, {labels.size} points ({(labels < 0).mean():.1%} noise)"
        )
        fig.tight_layout()
        return pd.Series(sizes).describe()

    print(cluster_size_distribution(clusterer, layer=0))
    plt.show()

    # ---- per-cluster gini of membership strength ----
    def gini(x):
        x = np.sort(np.asarray(x, dtype=np.float64))
        n = len(x)
        s = x.sum()
        if n < 2 or s == 0:
            return 0.0
        return float((2.0 * np.arange(1, n + 1) @ x) / (n * s) - (n + 1) / n)

    def per_cluster_strength_gini(clusterer, layer):
        labels = clusterer.cluster_layers_[layer]
        strengths = clusterer.membership_strength_layers_[layer]
        rows = []
        for cid in np.unique(labels):
            if cid < 0:
                continue
            s = strengths[labels == cid]
            if len(s) < 2:
                continue
            rows.append({"cluster": int(cid), "size": int(len(s)), "gini": gini(s)})
        return pd.DataFrame(rows).sort_values("gini")

    gini_df = per_cluster_strength_gini(clusterer, layer=0)
    print(f"layer 0: {len(gini_df)} clusters")
    print(
        f"mean gini: {gini_df['gini'].mean():.3f}  (0 = all-equal/arbitrary argmax, →1 = peaked)"
    )
    print(gini_df["gini"].describe())

    def plot_strength_curves(clusterer, gini_df, layer, k=5, min_size=20):
        df = gini_df[gini_df["size"] >= min_size]
        low = df.nsmallest(k, "gini")
        high = df.nlargest(k, "gini")
        labels = clusterer.cluster_layers_[layer]
        strengths = clusterer.membership_strength_layers_[layer]
        fig, axes = plt.subplots(2, k, figsize=(3 * k, 5), sharey=True)
        for row_idx, (title, sel) in enumerate(
            [
                (f"lowest gini (near-uniform), size>={min_size}", low),
                (f"highest gini (peaked), size>={min_size}", high),
            ]
        ):
            for ax, (_, row) in zip(axes[row_idx], sel.iterrows()):
                s = np.sort(strengths[labels == row["cluster"]])[::-1]
                ax.plot(s, marker=".", markersize=3, linewidth=1)
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlabel("rank")
                ax.set_title(
                    f"cid={int(row['cluster'])} n={int(row['size'])}\ngini={row['gini']:.2f}"
                )
            axes[row_idx, 0].set_ylabel(title)
        fig.suptitle(f"Layer {layer} — strengths sorted descending (flat = degenerate)")
        fig.tight_layout()
        return fig

    plot_strength_curves(clusterer, gini_df, layer=0, k=5, min_size=20)
    plt.savefig("tmp/gini.png")
    plt.show()

    # ---- legacy cluster-exemplar dictionary builds (abandoned approach) ----
    def cluster_exemplar_idxs(clusterer, layers=None, only_in_tree=True, quantile=0.99):
        all_layers = clusterer.cluster_layers_
        all_strengths = clusterer.membership_strength_layers_
        if layers is None:
            layers_iter = range(len(all_layers))
            single = False
        elif isinstance(layers, int):
            layers_iter = [layers]
            single = True
        else:
            layers_iter = list(layers)
            single = False
        if only_in_tree:
            tree_nodes = set(clusterer.cluster_tree_.keys())
            for kids in clusterer.cluster_tree_.values():
                tree_nodes.update(kids)
        else:
            tree_nodes = None
        result = {}
        for layer_idx in layers_iter:
            labels = all_layers[layer_idx]
            strengths = all_strengths[layer_idx]
            for cid in np.unique(labels):
                if cid < 0:
                    continue
                if tree_nodes is not None and (layer_idx, int(cid)) not in tree_nodes:
                    continue
                idxs = np.where(labels == cid)[0]
                cluster_strengths = strengths[idxs]
                threshold = np.quantile(cluster_strengths, quantile)
                top = idxs[cluster_strengths >= threshold]
                if single:
                    result[int(cid)] = top
                else:
                    result[(layer_idx, int(cid))] = top
        return result

    def cluster_exemplars(
        clusterer,
        embeddings,
        layers=None,
        only_in_tree=True,
        normalize=True,
        quantile=0.99,
    ):
        exemplar_indices = cluster_exemplar_idxs(
            clusterer, layers=layers, only_in_tree=only_in_tree, quantile=quantile
        )
        exemplar_embeddings = np.stack(
            [embeddings[idxs].mean(axis=0) for idxs in exemplar_indices.values()]
        )
        if normalize:
            exemplar_embeddings = (
                exemplar_embeddings
                / np.linalg.norm(exemplar_embeddings, axis=1)[:, np.newaxis]
            )
        return exemplar_embeddings

    lowlevel_dictionary = cluster_exemplars(clusterer, pythia_embeddings, layers=[0])
    print("lowlevel_dictionary.shape:", lowlevel_dictionary.shape)
    with open("leaf_exemplars_pythia70m_layer3_n10000.npy", "wb") as f:
        np.save(f, lowlevel_dictionary)
    n_layers = len(clusterer.cluster_layers_)
    nonterminal_dictionary = cluster_exemplars(
        clusterer, pythia_embeddings, layers=range(1, n_layers)
    )
    print("nonterminal_dictionary.shape:", nonterminal_dictionary.shape)
    with open("nonterminal_exemplars_pythia70m_layer3_n10000.npy", "wb") as f:
        np.save(f, nonterminal_dictionary)
    almost_nonterminal_dictionary = cluster_exemplars(
        clusterer, pythia_embeddings, layers=range(1, n_layers - 1)
    )
    with open("almost_nonterminal_exemplars_pythia70m_layer3_n10000.npy", "wb") as f:
        np.save(f, almost_nonterminal_dictionary)
    print("almost_nonterminal_dictionary.shape:", almost_nonterminal_dictionary.shape)

# %% [markdown]
# ## GMRA - geometric wavelets from a hierarchical partition
#
# Build a multiscale wavelet basis on top of the hierarchy returned by
# ``build_hierarchy`` above and use it as a sparse encoder/decoder. Each
# sample gets routed (adaptively, by greedy residual descent) to a stop
# node and represented by tree-height-many wavelet blocks along its
# root→stop chain.

# %%
# %%time
gmra = GMRA(
    hierarchy=hierarchy,
    basis_dim_strategy=FixedManifoldDim(k=32),
    max_dim=64,
    threshold=0.1,
).fit(pythia_embeddings)

# %%
print(f"wavelets shape: {gmra.wavelets.shape}")
print(f"n_atoms: {gmra.n_atoms}")

# %% [markdown]
# ### Adaptive transform — two criteria
#
# Each sample greedily descends the tree, stopping at the first node where
# no child reduces the routing residual under the chosen criterion. This
# replaces the old nearest-leaf Voronoi routing with sample-specific depth.

# %%
# %%time
codes_center, paths_center = gmra.transform(
    pythia_embeddings, adaptive=True, criterion="center"
)

# %%
# %%time
codes_proj, paths_proj = gmra.transform(
    pythia_embeddings, adaptive=True, criterion="projection"
)


# %%
def depth_stats(paths, label):
    depths = np.array([len(p) - 1 for p in paths])  # 0 == stopped at root
    print(f"== adaptive transform ({label}) ==")
    print(f"  samples: {len(paths)}")
    print(
        f"  stop-depth: min={depths.min()}, mean={depths.mean():.2f}, "
        f"median={int(np.median(depths))}, max={depths.max()}"
    )
    print(f"  fraction stopped at root: {(depths == 0).mean():.3%}")
    print(f"  fraction stopped at depth 1: {(depths == 1).mean():.3%}")
    print(f"  fraction descended past depth 3: {(depths > 3).mean():.3%}")
    return depths


depths_center = depth_stats(paths_center, "criterion=center")
print()
depths_proj = depth_stats(paths_proj, "criterion=projection")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, d, label in [
    (axes[0], depths_center, "criterion=center"),
    (axes[1], depths_proj, "criterion=projection"),
]:
    bins = np.arange(d.min(), d.max() + 2) - 0.5
    ax.hist(d, bins=bins, alpha=0.8)
    ax.set_xlabel("stop depth (0 = root)")
    ax.set_ylabel("# samples")
    ax.set_title(f"{label} — n_atoms_per_sample varies")
fig.suptitle("Distribution of adaptive stop depth")
fig.tight_layout()
plt.savefig("tmp/adaptive_depth_hist.png", dpi=100, bbox_inches="tight")
plt.show()


# %%
def codes_summary(codes, paths, label):
    nnz_per_row = np.diff(codes.indptr)  # csr indptr trick
    print(f"== codes ({label}) ==")
    print(f"  shape: {codes.shape}")
    print(f"  total nnz: {codes.nnz}")
    print(f"  mean nnz per row: {nnz_per_row.mean():.1f}")
    print(f"  median nnz per row: {int(np.median(nnz_per_row))}")
    print(f"  sparsity: {1 - codes.nnz / (codes.shape[0] * codes.shape[1]):.6f}")


codes_summary(codes_center, paths_center, "center")
print()
codes_summary(codes_proj, paths_proj, "projection")


# %%
def streaming_reconstruction_stats(X, codes, paths, gmra, chunk: int = 10_000):
    """Stream over rows so we never materialize (n_samples, d)-sized temps."""
    n = X.shape[0]
    sse = 0.0
    sum_x = 0.0
    sum_x2 = 0.0
    for i in range(0, n, chunk):
        x = X[i : i + chunk]
        x_hat = gmra.inverse_transform(codes[i : i + chunk], paths[i : i + chunk])
        diff = x_hat - x
        sse += float(np.einsum("ij,ij->", diff, diff))
        sum_x += float(x.sum(dtype=np.float64))
        sum_x2 += float(np.einsum("ij,ij->", x, x))
    n_elem = n * X.shape[1]
    mse = sse / n_elem
    mean_x = sum_x / n_elem
    var = sum_x2 / n_elem - mean_x**2
    return mse, var


# %%
# %%time
mse_center, var_x = streaming_reconstruction_stats(
    pythia_embeddings, codes_center, paths_center, gmra
)
mse_proj, _ = streaming_reconstruction_stats(
    pythia_embeddings, codes_proj, paths_proj, gmra
)
print(f"data variance:                          {var_x:.6f}")
print(
    f"reconstruction MSE (center criterion):     {mse_center:.6f}   MSE/Var = {mse_center / var_x:.4f}"
)
print(
    f"reconstruction MSE (projection criterion): {mse_proj:.6f}   MSE/Var = {mse_proj / var_x:.4f}"
)

# Adopt the projection-criterion output as the canonical pair for downstream cells.
# Override the variable name `leaves` (kept for backwards compat with later cells)
# to be the paths from the projection criterion.
codes = codes_proj
leaves = (
    paths_proj  # NOTE: variable name retained for backwards-compat; semantics changed.
)

# %% [markdown]
# ## Interpretability — best-fit examples per node
#
# For a node N, we score samples routed through it by their perpendicular
# distance to N's local affine subspace `c_N + span(Φ_N)`. Low residual =
# x sits close to N's hyperplane = x is representative of N's local linear
# model. We pick the top-level nodes (root + its children) and 5 random
# leaves to visualize.

# %%
from gmra_interp import best_fit_per_node, samples_in_subtree
import matplotlib.pyplot as plt


def compute_node_depth(hierarchy):
    """Map node_id -> depth-from-root in the GMRA tree. Root is depth 0."""
    depth_of = {}
    stack = [(hierarchy.root, 0)]
    while stack:
        node, d = stack.pop()
        depth_of[node.node_id] = d
        for c in node.children:
            stack.append((c, d + 1))
    return depth_of


def node_label(node_id, depth_of):
    """Label in GMRA-tree terms. node_id is shown as a tag — its meaning
    depends on TREE_TYPE: ``(evoc_layer, cluster_id)`` for EVoC,
    ``(tree_depth, ordinal_at_depth)`` for dyadic."""
    d = depth_of[node_id]
    return f"root [id={node_id}]" if d == 0 else f"GMRA depth {d} [id={node_id}]"


depth_of = compute_node_depth(hierarchy)

# %%
# Pick GMRA top-level nodes: the root + the K largest of its direct children
# (depth-1 nodes ranked by subtree size). The root often has many tiny
# orphan-children at various EVoC layers; we drop those.
root_node = hierarchy.root

children_by_subtree_size = sorted(
    root_node.children,
    key=lambda c: samples_in_subtree(gmra, leaves, c.node_id).size,
    reverse=True,
)
K_DEPTH1 = 6
top_level_ids = [root_node.node_id] + [
    c.node_id for c in children_by_subtree_size[:K_DEPTH1]
]

rng = np.random.default_rng(0)
n_leaf_samples = min(20, len(gmra.leaf_ids))
sampled_leaf_ids = [
    gmra.leaf_ids[i]
    for i in rng.choice(len(gmra.leaf_ids), size=n_leaf_samples, replace=False)
]

print(f"root has {len(root_node.children)} direct children (GMRA depth 1)")
print(f"keeping top {K_DEPTH1} by subtree size:")
for c in children_by_subtree_size[:K_DEPTH1]:
    n_routed = samples_in_subtree(gmra, leaves, c.node_id).size
    print(f"  {node_label(c.node_id, depth_of)} — {n_routed} samples routed")
print(f"\nsampled leaf nodes (GMRA leaf depth varies):")
for lid in sampled_leaf_ids:
    print(f"  {node_label(lid, depth_of)}")

# %%
center_only = True

# %%
# %%time
TOP_K = 10
best_fit_top = best_fit_per_node(
    pythia_embeddings,
    gmra,
    leaves,
    top_k=TOP_K,
    node_ids=top_level_ids,
    center_only=center_only,
)
best_fit_leaves = best_fit_per_node(
    pythia_embeddings,
    gmra,
    leaves,
    top_k=TOP_K,
    node_ids=sampled_leaf_ids,
    center_only=center_only,
)


# %%
def plot_residuals_per_node(
    X,
    gmra,
    leaf_assignments,
    results,
    title,
    center_only: bool,
    max_hist_samples=20_000,
    rng=None,
):
    """For each node in `results`, plot the residual distribution over its
    routed samples with the top-k residuals overlaid as a rug.

    ``center_only`` MUST match the value passed to ``best_fit_per_node`` that
    produced ``results`` — otherwise the rug is in different units than the
    histogram and the marks will look like outliers.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    metric_label = "‖x − c‖" if center_only else "‖(I − ΦᵀΦ)(x − c)‖"
    n_nodes = len(results)
    fig, axes = plt.subplots(n_nodes, 1, figsize=(8, 2.4 * n_nodes), squeeze=False)
    for ax, (node_id, (top_idxs, top_res)) in zip(axes[:, 0], results.items()):
        Phi = gmra.basis[node_id]
        c = gmra.center[node_id]
        subtree_idxs = samples_in_subtree(
            gmra, leaf_assignments, node_id, limit=max_hist_samples, rng=rng
        )
        from gmra_interp import _chunked_residuals

        residuals = np.sqrt(
            _chunked_residuals(
                X[subtree_idxs], c, Phi, center_only=center_only, chunk=10_000
            )
        )
        top_dist = np.sqrt(top_res)
        ax.hist(residuals, bins=60, alpha=0.7)
        ax.set_yscale("log")
        ax.set_xlabel(f"distance {metric_label}")
        ax.set_ylabel("# samples (log)")
        ax.set_title(
            f"{node_label(node_id, depth_of)} — basis dim {Phi.shape[0]}, "
            f"subtree {samples_in_subtree(gmra, leaf_assignments, node_id).size} samples — "
            f"top-{len(top_dist)} distances: min={top_dist.min():.3g}, max={top_dist.max():.3g}"
        )
        for r in top_dist:
            ax.axvline(r, color="red", alpha=0.5, linewidth=1)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# %%
plot_residuals_per_node(
    pythia_embeddings,
    gmra,
    leaves,
    best_fit_top,
    title="Top-level nodes — distance distribution + top-10 marked",
    center_only=center_only,
)
plt.savefig("tmp/best_fit_top_level.png", dpi=100, bbox_inches="tight")
plt.show()

# %%
plot_residuals_per_node(
    pythia_embeddings,
    gmra,
    leaves,
    best_fit_leaves,
    title="Sampled leaf nodes — distance distribution + top-10 marked",
    center_only=center_only,
)
plt.savefig("tmp/best_fit_leaves.png", dpi=100, bbox_inches="tight")
plt.show()

# %%
pd.Series(np.linalg.norm(pythia_embeddings, axis=1)).describe()

# %%
# Print top samples per node for closer inspection.
print("== top-level GMRA nodes (root + largest depth-1 children) ==")
for node_id, (idxs, res) in best_fit_top.items():
    print(f"\n{node_label(node_id, depth_of)}: top {len(idxs)} sample idxs / distances")
    for i, r in zip(idxs, np.sqrt(res)):
        print(f"  sample {i:8d}: distance {r:.4g}")


# %%
print("\n== sampled GMRA leaves ==")
for node_id, (idxs, res) in best_fit_leaves.items():
    print(f"\n{node_label(node_id, depth_of)}: top {len(idxs)} sample idxs / distances")
    for i, r in zip(idxs, np.sqrt(res)):
        print(f"  sample {i:8d}: distance {r:.4g}")

# %% [markdown]
# ## Max-activating examples — decoded token context per node
#
# For each node, surface the K samples this node fits best (smallest
# residual against the local affine subspace c_N + span(Φ_N)) and decode
# their surrounding tokens via the Pythia tokenizer. Same residual
# semantics as best_fit_per_node; the added value is the token-level
# context for human reading.

# %%
import importlib
import gmra_interp

importlib.reload(gmra_interp)
from gmra_interp import GMRATokenInterp


# %%
from gmra_interp import GMRATokenInterp

interp = GMRATokenInterp(
    gmra=gmra,
    X=pythia_embeddings,
    paths=paths_proj,
    df=pythia_embeddings_df,
    model_name="EleutherAI/pythia-70m",
)

# %%
# rich's Console auto-detects the Jupyter frontend and renders ANSI codes
# as coloured spans there; in a terminal it just emits the escape codes.
from rich.console import Console
from rich.text import Text

console = Console()


def print_examples(nid):
    console.print(f"\n{'=' * 72}\n{node_label(nid, depth_of)}\n{'=' * 72}")
    examples = interp.max_activating_examples(nid, k=TOP_K_EXAMPLES, context_size=5)
    for _, row in examples.iterrows():
        # from_ansi parses the focus-token's ANSI red wrapping; the surrounding
        # bracket-text is treated as literal characters, not Rich markup.
        console.print(
            Text.from_ansi(f"  [{np.sqrt(row.residual):.3g}] {row.context_text}")
        )


# %%
TOP_K_EXAMPLES = 10
for nid in top_level_ids:
    print_examples(nid)

# %%
for nid in sampled_leaf_ids:
    print_examples(nid)

# %% [markdown]
# ### Sanity check: residuals along the adaptive root → stop path
#
# With adaptive routing, the routing-criterion residual is monotonically
# nonincreasing along each sample's path **by construction** — we only
# descend when the next node reduces it. We trace residuals here both
# under the criterion used for routing and under the other metric, for
# 12 random samples.
#
# Expected result:
#  - Center-criterion paths: center residual nonincreasing; projection
#    residual may fluctuate.
#  - Projection-criterion paths: projection residual nonincreasing; center
#    residual may fluctuate.

# %%
from gmra_interp import path_residuals

# %%
n_trace = 32
trace_idxs = rng.choice(pythia_embeddings.shape[0], size=n_trace, replace=False)

# Traces under the projection-criterion paths.
trace_proj_paths_projres = path_residuals(
    pythia_embeddings, gmra, paths_proj, trace_idxs, center_only=False
)
trace_proj_paths_centres = path_residuals(
    pythia_embeddings, gmra, paths_proj, trace_idxs, center_only=True
)
# Traces under the center-criterion paths.
trace_cent_paths_projres = path_residuals(
    pythia_embeddings, gmra, paths_center, trace_idxs, center_only=False
)
trace_cent_paths_centres = path_residuals(
    pythia_embeddings, gmra, paths_center, trace_idxs, center_only=True
)

# %%
pd.Series(np.linalg.norm(pythia_embeddings, axis=1)).describe()

# %%
np.linalg.norm(pythia_embeddings, axis=1)[trace_idxs[0]]

# %%
(_, res) = trace_cent_paths_centres[0]

# %%
np.sqrt(res)

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
panels = [
    (
        axes[0, 0],
        trace_cent_paths_centres,
        "center-criterion path • center residual (must decrease)",
    ),
    (
        axes[0, 1],
        trace_cent_paths_projres,
        "center-criterion path • projection residual",
    ),
    (
        axes[1, 0],
        trace_proj_paths_centres,
        "projection-criterion path • center residual",
    ),
    (
        axes[1, 1],
        trace_proj_paths_projres,
        "projection-criterion path • projection residual (must decrease)",
    ),
]
for ax, traces, label in panels:
    for sample_idx, (_, res) in zip(trace_idxs, traces):
        ax.plot(
            range(len(res)),
            np.sqrt(res),
            marker="o",
            alpha=0.7,
            label=f"sample {sample_idx} (depth {len(res) - 1})",
        )
    ax.set_xlabel("depth from root")
    ax.set_ylabel("residual")
    ax.set_title(label)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=7, loc="upper right")
fig.suptitle("Residuals along adaptive paths — 12 random samples")
fig.tight_layout()
plt.savefig("tmp/path_residuals.png", dpi=100, bbox_inches="tight")
plt.show()


# %%
def monotonicity_score(residuals: np.ndarray) -> float:
    if len(residuals) < 2:
        return float("nan")
    return float(np.mean(residuals[1:] <= residuals[:-1] + 1e-8))


print(
    "Monotonic-decrease fraction along adaptive paths (1.0 = strictly nonincreasing):"
)
print(
    f"{'sample':>10}  {'depth(c)':>9}  {'depth(p)':>9}  "
    f"{'C-path/C-res':>13}  {'C-path/P-res':>13}  {'P-path/C-res':>13}  {'P-path/P-res':>13}"
)
for k, sample_idx in enumerate(trace_idxs):
    d_c = len(trace_cent_paths_centres[k][1]) - 1
    d_p = len(trace_proj_paths_projres[k][1]) - 1
    mc_cc = monotonicity_score(trace_cent_paths_centres[k][1])
    mc_cp = monotonicity_score(trace_cent_paths_projres[k][1])
    mc_pc = monotonicity_score(trace_proj_paths_centres[k][1])
    mc_pp = monotonicity_score(trace_proj_paths_projres[k][1])
    print(
        f"{sample_idx:>10d}  {d_c:>9d}  {d_p:>9d}  "
        f"{mc_cc:>13.2f}  {mc_cp:>13.2f}  {mc_pc:>13.2f}  {mc_pp:>13.2f}"
    )

# %%
