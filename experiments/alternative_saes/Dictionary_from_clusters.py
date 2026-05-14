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
pythia_embeddings = col.values.to_numpy().reshape(-1, 512)

# %%
# %%time

pythia_embeddings = np.unique(pythia_embeddings, axis=0)

# %%
pythia_embeddings = pythia_embeddings - pythia_embeddings.mean(axis=0)
pythia_embeddings = (
    pythia_embeddings / np.linalg.norm(pythia_embeddings, axis=1)[:, np.newaxis]
)

# %%
pythia_embeddings.shape

# %%
# %%time

clusterer = evoc.EVoC(
    noise_level=0.5, max_layers=10, n_epochs=100, base_min_cluster_size=25
)  # , base_n_clusters=10000)
cluster_labels = clusterer.fit_predict(pythia_embeddings)

# %%
pd.Series(cluster_labels).value_counts()

# %%
import numpy as np

# How big is the tree really?
tree = clusterer.cluster_tree_
all_tree_nodes = set(tree.keys())
for kids in tree.values():
    all_tree_nodes.update(kids)

print(f"Total nodes in tree: {len(all_tree_nodes)}")
print(f"Tree keys (parents): {len(tree)}")
print(f"Total edges (parent->child): {sum(len(v) for v in tree.values())}")

# Per-layer breakdown
from collections import Counter

nodes_per_layer = Counter(n[0] for n in all_tree_nodes)
print(f"\nTree nodes per layer: {dict(sorted(nodes_per_layer.items()))}")

# Compare to label vectors
print(f"\nUnique cluster IDs per layer (from cluster_layers_):")
for i, labels in enumerate(clusterer.cluster_layers_):
    n_clusters = len(np.unique(labels[labels >= 0]))
    print(f"  Layer {i}: {n_clusters}")

# %% [markdown]
# ## Creating a dictionary - cluster exemplars

# %%
pd.Series(clusterer.membership_strength_layers_[0]).quantile(0.9)

# %%
clusterer.membership_strength_layers_[0].shape

# %% [markdown]
# ### Cluster size distribution (layer 0)
#
# Context for the diagnostics below — argmax-based exemplar selection
# behaves very differently for tiny vs large clusters, so we need to know
# the size distribution before reading any per-cluster metric.

# %%
import matplotlib.pyplot as plt


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


cluster_size_distribution(clusterer, layer=0)

# %% [markdown]
# ### Diagnostic: membership strength distribution

# %%
import numpy as np


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


# %%
gini_df = per_cluster_strength_gini(clusterer, layer=0)
print(f"layer 0: {len(gini_df)} clusters")
print(
    f"mean gini: {gini_df['gini'].mean():.3f}  (0 = all-equal/arbitrary argmax, →1 = peaked)"
)
gini_df["gini"].describe()

# %%
import matplotlib.pyplot as plt


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

# %%


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


# %%
def cluster_exemplars(
    clusterer, embeddings, layers=None, only_in_tree=True, normalize=True, quantile=0.99
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


# %%
lowlevel_dictionary = cluster_exemplars(clusterer, pythia_embeddings, layers=[0])

# %%
lowlevel_dictionary.shape

# %%
with open("leaf_exemplars_pythia70m_layer3_n10000.npy", "wb") as f:
    np.save(f, lowlevel_dictionary)

# %%
n_layers = len(clusterer.cluster_layers_)

# %%
nonterminal_dictionary = cluster_exemplars(
    clusterer, pythia_embeddings, layers=range(1, n_layers)
)

# %%
nonterminal_dictionary.shape

# %%
with open("nonterminal_exemplars_pythia70m_layer3_n10000.npy", "wb") as f:
    np.save(f, nonterminal_dictionary)

# %%
almost_nonterminal_dictionary = cluster_exemplars(
    clusterer, pythia_embeddings, layers=range(1, n_layers - 1)
)

# %%
with open("almost_nonterminal_exemplars_pythia70m_layer3_n10000.npy", "wb") as f:
    np.save(f, almost_nonterminal_dictionary)

# %%
almost_nonterminal_dictionary.shape

# %%
almost_nonterminal_dictionary = cluster_exemplars(
    clusterer, pythia_embeddings, layers=range(1, n_layers - 1)
)

# %%
toplevel_dictionary = cluster_exemplars(
    clusterer, pythia_embeddings, layers=[n_layers - 1]
)

# %%
all_dictionary = cluster_exemplars(clusterer, pythia_embeddings)

# %%
with open("all_exemplars_pythia70m_layer3_n10000.npy", "wb") as f:
    np.save(f, all_dictionary)


# %% [markdown]
# ## Mutual coherence: per-layer vs full dictionary
#
# Mutual coherence of a dictionary D with unit-norm atoms d_i is
# mu(D) = max_{i != j} |<d_i, d_j>|. Low coherence is desirable for
# compressive-sensing-style recovery guarantees (e.g. mu < 1/(2k-1)
# implies unique k-sparse recovery via BP/OMP).
#
# We compare the coherence of the per-layer atom matrices against the
# coherence of the concatenation across all layers — the latter is
# expected to be larger because cross-layer atoms can be near-duplicates
# (a fine cluster and its parent often share an exemplar direction).


# %%
def mutual_coherence(D):
    # D: (n_atoms, dim), assumed to have unit-norm rows
    if D.shape[0] < 2:
        return np.nan
    G = D @ D.T
    np.fill_diagonal(G, 0.0)
    return float(np.max(np.abs(G)))


def compare_coherence(clusterer, embeddings):
    per_layer = {
        l: cluster_exemplars(clusterer, embeddings, layers=[l])
        for l in range(len(clusterer.cluster_layers_))
    }
    rows = [
        {"layer": l, "n_atoms": D.shape[0], "mutual_coherence": mutual_coherence(D)}
        for l, D in per_layer.items()
    ]
    full = np.concatenate(list(per_layer.values()), axis=0)
    rows.append(
        {
            "layer": "all (concat)",
            "n_atoms": full.shape[0],
            "mutual_coherence": mutual_coherence(full),
        }
    )
    return pd.DataFrame(rows)


# %%
compare_coherence(clusterer, pythia_embeddings)

# %% [markdown]
# ### Why is coherence still 1? — worst-pair diagnostic
#
# Coherence = 1 means two atoms are *identical* after normalization, not
# merely close. Possible causes:
#   (a) duplicate rows in `pythia_embeddings` — two clusters with singleton
#       exemplars at duplicate vectors give the same atom;
#   (b) two clusters' top-quantile point sets overlap on a dominant point;
#   (c) two clusters' averages happen to coincide (very unlikely by chance).
#
# We surface the offending pair so we can tell which.


# %%
def diagnose_worst_pair(clusterer, embeddings, layer, quantile=0.99):
    idx_map = cluster_exemplar_idxs(clusterer, layers=[layer], quantile=quantile)
    keys = list(idx_map.keys())
    D = cluster_exemplars(clusterer, embeddings, layers=[layer], quantile=quantile)
    G = D @ D.T
    np.fill_diagonal(G, 0.0)
    i, j = np.unravel_index(np.argmax(np.abs(G)), G.shape)
    ki, kj = keys[i], keys[j]
    idxs_i, idxs_j = idx_map[ki], idx_map[kj]
    overlap = set(idxs_i.tolist()) & set(idxs_j.tolist())
    return {
        "layer": layer,
        "max_coherence": float(G[i, j]),
        "cluster_i": ki,
        "size_i": len(idxs_i),
        "sample_idxs_i": idxs_i[:5].tolist(),
        "cluster_j": kj,
        "size_j": len(idxs_j),
        "sample_idxs_j": idxs_j[:5].tolist(),
        "overlap_point_idxs": list(overlap)[:5],
        "embeddings_equal": bool(
            np.allclose(embeddings[idxs_i].mean(0), embeddings[idxs_j].mean(0))
        ),
    }


for L in range(len(clusterer.cluster_layers_)):
    print(diagnose_worst_pair(clusterer, pythia_embeddings, layer=L))

# %%
# Dataset-level duplicate check — hypothesis (a)
unique_count = len(np.unique(pythia_embeddings, axis=0))
print(
    f"{unique_count} unique / {len(pythia_embeddings)} total embeddings ({len(pythia_embeddings) - unique_count} duplicates)"
)

# %%
