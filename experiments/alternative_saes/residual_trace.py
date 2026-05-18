"""Trace residuals along adaptive GMRA paths — how they change between levels."""

import numpy as np
import logging
from tqdm.auto import tqdm
from gmra import GMRA, FixedManifoldDim
from tree_adapters import build_hierarchy
from gmra_interp import (
    path_residuals,
    samples_in_subtree,
    best_fit_per_node,
    _chunked_residuals,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data…")
import pandas as pd

df = pd.read_parquet("pythia70m_layer3_n10000.parquet")
col = df["activation"].values.reshape(-1, 512)
_, idx = np.unique(col, axis=0, return_index=True)
X = col[idx].astype(np.float32)
df = df.iloc[idx].drop(columns=["activation"]).reset_index(drop=True)
print(f"  {X.shape[0]:,} unique samples, {X.shape[1]} dims")

# ── 2. Build hierarchy (dyadic) ───────────────────────────────────────────────
print("Building dyadic hierarchy…")
hierarchy, clusterer = build_hierarchy(
    "dyadic",
    X=X,
    evoc_kwargs={},
    dyadic_max_depth=12,
    dyadic_k=50,
    dyadic_min_cell_size=25,
)
print(
    f"  height={hierarchy.height}, leaves={len(hierarchy.leaves)}, nodes={len(hierarchy.nodes_by_id)}"
)

# ── 3. Fit GMRA ───────────────────────────────────────────────────────────────
print("Fitting GMRA…")
gmra = GMRA(
    hierarchy=hierarchy,
    basis_dim_strategy=FixedManifoldDim(k=32),
    max_dim=64,
    threshold=0.1,
).fit(X)
print(f"  wavelets shape: {gmra.wavelets.shape}, n_atoms: {gmra.n_atoms}")

# ── 4. Adaptive transform (projection criterion) ─────────────────────────────
print("Adaptive transform (projection)…")
codes, paths = gmra.transform(X, adaptive=True, criterion="projection")

# ── 5. Depth statistics ───────────────────────────────────────────────────────
depths = np.array([len(p) - 1 for p in paths])
print(
    f"\nStop-depth stats: min={depths.min()}, mean={depths.mean():.2f}, "
    f"median={int(np.median(depths))}, max={depths.max()}"
)
for d in range(0, depths.max() + 1):
    frac = (depths == d).mean()
    if frac > 0.005:
        print(f"  depth {d}: {(depths == d).sum():6d} samples ({frac:.1%})")

# ── 6. Compute ALL residuals efficiently in one batch call ────────────────────
print("\nComputing residuals for all samples…")
all_paths = [p for p in paths]
all_residuals_sq = np.zeros((len(paths), max(len(p) for p in all_paths)))

for i in tqdm(range(len(X)), desc="residuals", unit="sample"):
    x = X[i : i + 1]
    path = all_paths[i]
    sq = _chunked_residuals(
        x, gmra.center[path[0]], gmra.basis[path[0]], center_only=False, chunk=10000
    )
    for step, nid in enumerate(path):
        if step == 0:
            r = sq[0]
        else:
            diff = x - gmra.center[nid]
            Phi = gmra.basis[nid]
            diff_proj = diff - diff @ Phi.T @ Phi
            r = float((diff_proj * diff_proj).sum())
        all_residuals_sq[i, step] = r

all_residuals = np.sqrt(all_residuals_sq)  # L2 residuals

# ── 7. Aggregate: mean residual per depth level ───────────────────────────────
print(f"\n{'=' * 80}")
print("Aggregated: mean ± std residual per depth level (all samples)")
print(f"{'=' * 80}\n")

max_depth = all_residuals_sq.shape[1] - 1
for d in range(max_depth + 1):
    vals = all_residuals[:, d][np.isfinite(all_residuals[:, d])]
    if len(vals) == 0:
        continue
    print(
        f"  depth {d:>2d}: n={len(vals):6d}  mean={vals.mean():8.4f}  "
        f"median={np.median(vals):8.4f}  std={vals.std():8.4f}  "
        f"min={vals.min():8.4f}  max={vals.max():8.4f}"
    )

# ── 8. Fraction of variance explained per depth ───────────────────────────────
print(f"\n{'=' * 80}")
print("Fraction of data variance explained at each depth")
print(f"{'=' * 80}\n")

total_var = np.var(X, axis=0).sum()
for d in range(max_depth + 1):
    vals = all_residuals[:, d][np.isfinite(all_residuals[:, d])]
    if len(vals) == 0:
        continue
    mse_at_d = (vals**2).mean()
    frac = 1 - mse_at_d / total_var
    print(
        f"  depth {d:>2d}: MSE={mse_at_d:.6f}  var explained={frac:.4f} "
        f"({frac * 100:.1f}%)"
    )

# ── 9. Monotonicity check ────────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print("Monotonicity: fraction of steps where residual decreases (should ≈ 1.0)")
print(f"{'=' * 80}\n")

decrease_count = 0
total_steps = 0
for i in range(len(paths)):
    r = all_residuals[i]
    for j in range(1, len(r)):
        total_steps += 1
        if r[j] <= r[j - 1] + 1e-8:
            decrease_count += 1

print(
    f"  {decrease_count}/{total_steps} = {decrease_count / total_steps:.4f} "
    f"monotonically nonincreasing"
)

# ── 10. Trace a few random samples ───────────────────────────────────────────
rng = np.random.default_rng(42)
n_trace = 10
trace_idxs = rng.choice(X.shape[0], size=n_trace, replace=False)

print(f"\n{'=' * 80}")
print("Residual traces along adaptive paths (projection criterion)")
print(f"{'=' * 80}")
print()

for k, si in enumerate(trace_idxs):
    r = all_residuals[si]
    path = all_paths[si]
    depth = len(path) - 1
    print(f"Sample {si:5d}  (depth={depth})")
    for step, nid in enumerate(path):
        sub_size = samples_in_subtree(gmra, paths, nid).size
        r_val = np.sqrt(r[step]) if np.isfinite(r[step]) else float("inf")
        print(
            f"  step {step:>2d}  node {nid!s:>30s}  residual={r_val:8.4f}  "
            f"subtree_size={sub_size:5d}"
        )
    if k < n_trace - 1:
        print()

# ── 11. Best-fit examples at top-level nodes ─────────────────────────────────
print(f"\n{'=' * 80}")
print("Top-3 best-fit samples per top-level node (smallest residual)")
print(f"{'=' * 80}\n")

root = hierarchy.root
children_by_size = sorted(
    root.children,
    key=lambda c: samples_in_subtree(gmra, paths, c.node_id).size,
    reverse=True,
)
top_ids = [root.node_id] + [c.node_id for c in children_by_size[:3]]

results = best_fit_per_node(
    X, gmra, paths, top_k=3, node_ids=top_ids, center_only=False
)

for nid in top_ids:
    subsz = samples_in_subtree(gmra, paths, nid).size
    print(f"Node {nid!s:>30s}  (subtree_size={subsz})")
    idxs, res = results[nid]
    for si_local, ri in zip(idxs, np.sqrt(res)):
        global_si = (
            idxs[si_local] if hasattr(idxs, "__getitem__") else int(idxs[si_local])
        )
        print(f"  sample {global_si:5d}  residual={ri:.4f}")
    print()
