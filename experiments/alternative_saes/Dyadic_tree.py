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
# # Dyadic tree diagnostics — does METIS bisection actually give 2^{-j} scales?
#
# The paper's ``‖x − P_{M_j}(x)‖ ≤ C · 2^{-2j}`` bound assumes dyadic cells
# whose geometric diameter shrinks by 2 per level. METIS bisects on graph
# cut weight, not geometric diameter — let's measure how close we get.
#
# Diagnostics (top to bottom):
# 1. **kNN distance distribution** — pure data property; characterises the
#    affinity-graph input METIS sees. Pulled once at K=50 and read off for
#    smaller ranks.
# 2. **Cell diameter per depth** vs the theoretical 2^{-j} reference curve.
# 3. **Cell size + bisection balance** per depth.
# 4. **Local SVD spectrum per cell + residual floor for rank-k local PCA.**
#    Measures in-cell linearity — independent of how the tree was built;
#    bounds the residual any rank-k GMRA basis can achieve at that node.
#
# Tree construction itself uses no SVD anywhere — METIS only sees graph
# cuts. SVD enters here purely as a measurement tool.

# %%
# %%time
import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# %%
pythia_embeddings_df = pd.read_parquet("pythia70m_layer3_n10000.parquet")

# %%
# %%time
t = pq.read_table("pythia70m_layer3_n10000.parquet")
col = t.column("activation").combine_chunks()
pythia_embeddings_orig = col.values.to_numpy().reshape(-1, 512)

# %%
# %%time
_, unique_idxs = np.unique(pythia_embeddings_orig, axis=0, return_index=True)
pythia_embeddings = pythia_embeddings_orig[unique_idxs].astype(np.float32, copy=False)
pythia_embeddings_df = (
    pythia_embeddings_df.iloc[unique_idxs]
    .drop(columns=["activation"])
    .reset_index(drop=True)
)
assert len(pythia_embeddings_df) == pythia_embeddings.shape[0]
print(
    f"pythia_embeddings: shape={pythia_embeddings.shape}, dtype={pythia_embeddings.dtype}"
)

# %%
DYADIC_MAX_DEPTH = 12
DYADIC_K = 50  # METIS bandwidth uses K//2 = 25th-nearest neighbor
DYADIC_MIN_CELL_SIZE = 25

# %% [markdown]
# ## Diagnostic 1 — kNN distance distribution
#
# METIS bisects the affinity graph built by
# ``dyadic_decomposition.build_knn_affinity_graph``. Its bandwidth is
# ``eps_i = dist(x_i, K-th-NN(x_i))`` with ``K = DYADIC_K // 2``. Below: the
# distance distribution at several ranks. We pull k=50 once and read off
# smaller ranks from the same neighbor graph — no extra cost.

# %%
# %%time
import pynndescent

nn = pynndescent.NNDescent(
    pythia_embeddings,
    n_neighbors=DYADIC_K + 1,
    metric="euclidean",
    random_state=42,
    n_jobs=-1,
)
_, knn_dist = nn.neighbor_graph  # (n, DYADIC_K+1); column 0 == self, distance 0

# %%
fig, ax = plt.subplots(figsize=(9, 5))
for K, color in [
    (5, "tab:blue"),
    (10, "tab:orange"),
    (25, "tab:green"),
    (50, "tab:red"),
]:
    d = knn_dist[:, K]
    ax.hist(
        d,
        bins=80,
        alpha=0.4,
        color=color,
        label=f"K={K} (median={np.median(d):.3g})",
    )
ax.set_xlabel("distance to K-th nearest non-self neighbor")
ax.set_ylabel("# points")
ax.set_title(
    f"kNN distance distribution "
    f"(n={pythia_embeddings.shape[0]}, d={pythia_embeddings.shape[1]})"
)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig("tmp/dyadic_knn_hist.png", dpi=100, bbox_inches="tight")
plt.show()

# %%
print("Quantiles of K-th-NN distance:")
for K in [5, 10, 25, 50]:
    d = knn_dist[:, K]
    print(
        f"  K={K:2d}: q10={np.quantile(d, 0.1):.3g}  "
        f"median={np.median(d):.3g}  q90={np.quantile(d, 0.9):.3g}"
    )

# %%
# Median + q10/q90 band over the full range K = 1..DYADIC_K, so we can see
# whether distance grows ~linearly with K (= uniform density) or sub-linearly
# (= clumpy / low-dim structure).
Ks_full = np.arange(1, DYADIC_K + 1)
med = np.median(knn_dist[:, 1:], axis=0)
q10 = np.quantile(knn_dist[:, 1:], 0.1, axis=0)
q90 = np.quantile(knn_dist[:, 1:], 0.9, axis=0)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(Ks_full, med, color="black", label="median")
ax.fill_between(Ks_full, q10, q90, alpha=0.3, label="10–90% quantile")
ax.axvline(
    DYADIC_K // 2,
    color="red",
    linestyle="--",
    alpha=0.7,
    label=f"DYADIC_K // 2 = {DYADIC_K // 2} (METIS bandwidth)",
)
ax.set_xlabel("K (rank of nearest neighbor)")
ax.set_ylabel("distance to K-th nearest neighbor")
ax.set_title("How fast does the K-th-NN distance grow with K?")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig("tmp/dyadic_knn_curve.png", dpi=100, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Build dyadic hierarchy

# %%
from tree_adapters import build_hierarchy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
)

# %%
# %%time
hierarchy, _ = build_hierarchy(
    "dyadic",
    X=pythia_embeddings,
    dyadic_max_depth=DYADIC_MAX_DEPTH,
    dyadic_k=DYADIC_K,
    dyadic_min_cell_size=DYADIC_MIN_CELL_SIZE,
)
print(
    f"n_samples: {hierarchy.n_samples}  height: {hierarchy.height}  "
    f"leaves: {len(hierarchy.leaves)}  total nodes: {len(hierarchy.nodes_by_id)}"
)


# %%
def compute_node_depth(hierarchy):
    depth_of = {}
    stack = [(hierarchy.root, 0)]
    while stack:
        node, d = stack.pop()
        depth_of[node.node_id] = d
        for c in node.children:
            stack.append((c, d + 1))
    return depth_of


depth_of = compute_node_depth(hierarchy)
depths_sorted = sorted({depth_of[nid] for nid in hierarchy.nodes_by_id})

# %% [markdown]
# ## Diagnostic 2 — Cell diameter per depth
#
# For each node, estimate diameter via the max distance among ``n_pairs``
# random point pairs in the cell. If METIS were producing geometric
# bisection, median diameter would halve per level — that's the dashed
# reference curve, anchored at the root.


# %%
def estimate_cell_diameter(X, idxs, rng, n_pairs=400):
    if idxs.size < 2:
        return 0.0
    ia = rng.choice(idxs, size=n_pairs)
    ib = rng.choice(idxs, size=n_pairs)
    keep = ia != ib
    if not keep.any():
        return 0.0
    d = np.linalg.norm(X[ia[keep]] - X[ib[keep]], axis=1)
    return float(d.max())


# %%
# %%time
rng = np.random.default_rng(0)
depth_to_diams = defaultdict(list)
for nid, node in hierarchy.nodes_by_id.items():
    depth_to_diams[depth_of[nid]].append(
        estimate_cell_diameter(pythia_embeddings, node.idxs, rng)
    )

# %%
medians = np.array([np.median(depth_to_diams[d]) for d in depths_sorted])
q10s = np.array([np.quantile(depth_to_diams[d], 0.1) for d in depths_sorted])
q90s = np.array([np.quantile(depth_to_diams[d], 0.9) for d in depths_sorted])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(depths_sorted, medians, marker="o", color="black", label="median diameter")
ax.fill_between(depths_sorted, q10s, q90s, alpha=0.2, label="10–90% quantile")
ax.plot(
    depths_sorted,
    medians[0] * (2.0 ** -np.array(depths_sorted)),
    linestyle="--",
    color="red",
    label="root_diam · 2^{-j} (theoretical)",
)
ax.set_yscale("log")
ax.set_xlabel("depth")
ax.set_ylabel("cell diameter (random-pair upper bound estimate)")
ax.set_title("Cell diameter vs depth — METIS bisection of Pythia layer 3")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig("tmp/dyadic_diameter_by_depth.png", dpi=100, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Diagnostic 3 — Cell size and bisection balance

# %%
depth_to_sizes = defaultdict(list)
for nid, node in hierarchy.nodes_by_id.items():
    depth_to_sizes[depth_of[nid]].append(node.idxs.size)

fig, ax = plt.subplots(figsize=(10, 4))
ax.boxplot(
    [depth_to_sizes[d] for d in depths_sorted],
    positions=depths_sorted,
    widths=0.6,
    showfliers=False,
)
ax.set_yscale("log")
ax.set_xlabel("depth")
ax.set_ylabel("cell size")
ax.set_title("Cell size distribution per depth")
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig("tmp/dyadic_size_by_depth.png", dpi=100, bbox_inches="tight")
plt.show()

# %%
balance_ratios = []
for node in hierarchy.nodes_by_id.values():
    if len(node.children) == 2:
        a, b = node.children[0].idxs.size, node.children[1].idxs.size
        if a + b > 0:
            balance_ratios.append(min(a, b) / (a + b))
balance_ratios = np.asarray(balance_ratios)
print(
    f"bisection balance min(a,b)/(a+b): "
    f"min={balance_ratios.min():.3f}  median={np.median(balance_ratios):.3f}  "
    f"mean={balance_ratios.mean():.3f}  max={balance_ratios.max():.3f}"
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(balance_ratios, bins=40)
ax.axvline(0.5, color="red", linestyle="--", label="perfect 50/50 split")
ax.set_xlabel("min(left, right) / (left + right)")
ax.set_ylabel("# splits")
ax.set_title("Bisection balance across all splits")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Diagnostic 4 — Local SVD spectrum + residual floor per depth
#
# For sampled cells at each depth, compute the local SVD of
# ``X[idxs] - mean``. ``σ_i / sqrt(n_cell)`` is the per-point std along
# the i-th principal direction. The leftover
# ``sqrt(Σ_{i>k} (σ_i / sqrt n)²)`` is the per-point RMS residual norm if
# we keep ``k`` components — a hard lower bound on what any rank-k local
# affine basis (including GMRA's) can achieve at that node, regardless of
# how the tree was built.


# %%
def local_svd_spectrum(X, idxs, max_rank=40):
    if idxs.size < 2:
        return np.zeros(0)
    X_sub = X[idxs]
    X_centered = X_sub - X_sub.mean(axis=0)
    rank = min(max_rank, idxs.size, X.shape[1])
    _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
    return s[:rank] / np.sqrt(idxs.size)


def residual_floor_vs_k(s):
    """Per-point RMS residual norm if we keep k of the local components,
    for k = 0..len(s). Operates on normalised singular values
    ``σ_i / sqrt(n_cell)``."""
    if s.size == 0:
        return np.zeros(0)
    energy = s**2
    total = float(energy.sum())
    leftover = np.concatenate([[total], total - np.cumsum(energy)])  # k = 0..len(s)
    return np.sqrt(np.maximum(leftover, 0.0))


# %%
# %%time
SAMPLE_PER_DEPTH = 5
MAX_RANK = 40
nodes_by_depth = defaultdict(list)
for nid, node in hierarchy.nodes_by_id.items():
    nodes_by_depth[depth_of[nid]].append(node)

spectra_by_depth = {}
for d in depths_sorted:
    cands = nodes_by_depth[d]
    chosen = rng.choice(
        len(cands), size=min(SAMPLE_PER_DEPTH, len(cands)), replace=False
    )
    spectra_by_depth[d] = [
        local_svd_spectrum(pythia_embeddings, cands[i].idxs, MAX_RANK) for i in chosen
    ]

# %%
cmap = plt.cm.viridis
fig, ax = plt.subplots(figsize=(9, 5))
for d in depths_sorted:
    color = cmap(d / max(1, max(depths_sorted)))
    for s in spectra_by_depth.get(d, []):
        if s.size == 0:
            continue
        ax.plot(np.arange(1, s.size + 1), s, color=color, alpha=0.5, linewidth=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("component rank i")
ax.set_ylabel("σ_i / sqrt(n_cell)  (per-point std along direction i)")
ax.set_title(
    "Local SVD spectra at sampled cells — color = depth (dark→light = shallow→deep)"
)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig("tmp/dyadic_local_svd.png", dpi=100, bbox_inches="tight")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(9, 5))
for d in depths_sorted:
    floors = [residual_floor_vs_k(s) for s in spectra_by_depth.get(d, []) if s.size > 0]
    if not floors:
        continue
    color = cmap(d / max(1, max(depths_sorted)))
    max_len = max(f.size for f in floors)
    padded = np.array(
        [np.concatenate([f, np.full(max_len - f.size, np.nan)]) for f in floors]
    )
    median_floor = np.nanmedian(padded, axis=0)
    ax.plot(
        np.arange(median_floor.size),
        median_floor,
        color=color,
        alpha=0.85,
        linewidth=1.5,
        label=f"depth {d}",
    )
ax.axvline(
    16,
    color="red",
    linestyle="--",
    alpha=0.5,
    label="FixedManifoldDim(k=16) from Dictionary_from_clusters",
)
ax.set_yscale("log")
ax.set_xlabel("k (kept components)")
ax.set_ylabel("per-point residual floor sqrt(Σ_{i>k} (σ_i/√n)²)")
ax.set_title("Achievable residual lower bound vs k, by depth")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig("tmp/dyadic_residual_floor.png", dpi=100, bbox_inches="tight")
plt.show()
