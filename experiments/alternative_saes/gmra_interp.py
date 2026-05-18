"""Interpretability helpers for fitted GMRA models.

All helpers take ``paths`` (the list-of-lists returned by ``gmra.transform``)
rather than a flat leaf-assignment vector. Each path is a root → stop-node
chain of node_ids. With ``adaptive=True``, those stop nodes vary per sample
and aren't always leaves.

Two complementary views of "what does this node capture":

- ``samples_in_subtree(...)``: routing view. Samples whose path *passes
  through* this node — i.e. they reached this node during descent.
- ``best_fit_per_node(...)``: best-fit view. Among those samples, which fit
  the node's local linear model best (smallest perpendicular distance to the
  affine subspace ``c_N + span(rows of Φ_N)``).
- ``GMRATokenInterp``: SAE-style max-activating-examples — same best-fit
  criterion, plus token-context decoding via a HuggingFace tokenizer.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from gmra import GMRA

if TYPE_CHECKING:
    import pandas as pd


def _path_passes_through(
    paths: list[list[tuple[int, int]]], node_id: tuple[int, int]
) -> np.ndarray:
    """Boolean mask over samples: True iff ``node_id`` appears on sample's path."""
    mask = np.zeros(len(paths), dtype=bool)
    for i, p in enumerate(paths):
        if node_id in p:
            mask[i] = True
    return mask


def samples_in_subtree(
    gmra: GMRA,
    paths: list[list[tuple[int, int]]],
    node_id: tuple[int, int],
    limit: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """All sample indices whose path passes through ``node_id``.

    If ``limit`` is set and the matched set is larger, return a sorted random
    subsample of size ``limit`` (deterministic if ``rng`` is provided).
    """
    sample_idxs = np.flatnonzero(_path_passes_through(paths, node_id))
    if limit is not None and sample_idxs.size > limit:
        if rng is None:
            rng = np.random.default_rng()
        chosen = rng.choice(sample_idxs, size=limit, replace=False)
        chosen.sort()
        return chosen
    return sample_idxs


def best_fit_per_node(
    X: np.ndarray,
    gmra: GMRA,
    paths: list[list[tuple[int, int]]],
    top_k: int = 10,
    node_ids: list[tuple[int, int]] | None = None,
    center_only: bool = False,
    chunk: int = 10_000,
    progress: bool = True,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Top-k samples per node by smallest residual against N's local model.

    Default residual: ``‖(I − Φ_Nᵀ Φ_N)(x − c_N)‖²`` — distance from x to N's
    affine subspace. With ``center_only=True``: ``‖x − c_N‖²``.

    Candidate samples for each node are those whose path passes through that
    node (i.e. ``samples_in_subtree``).
    """
    if node_ids is None:
        node_ids = list(gmra.hierarchy.nodes_by_id.keys())

    # Precompute per-node pass-through masks so we don't recompute paths for
    # every node — flatten paths into a (n_samples, max_depth) lookup once.
    sample_path_sets = [set(p) for p in paths]

    results: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    iterator = node_ids
    if progress:
        iterator = tqdm(node_ids, desc="best_fit_per_node", unit="node")

    for node_id in iterator:
        sample_idxs = np.array(
            [i for i, s in enumerate(sample_path_sets) if node_id in s],
            dtype=np.int64,
        )
        if sample_idxs.size == 0:
            continue

        Phi = gmra.basis[node_id]
        c = gmra.center[node_id]
        residuals = _chunked_residuals(
            X_sub=X[sample_idxs],
            c=c,
            Phi=Phi,
            center_only=center_only,
            chunk=chunk,
        )

        k = min(top_k, len(residuals))
        top = np.argpartition(residuals, k - 1)[:k]
        order = np.argsort(residuals[top])
        top = top[order]
        results[node_id] = (sample_idxs[top].copy(), residuals[top].copy())
    return results


def path_residuals(
    X: np.ndarray,
    gmra: GMRA,
    paths: list[list[tuple[int, int]]],
    sample_idxs: np.ndarray,
    center_only: bool = False,
) -> list[tuple[list[tuple[int, int]], np.ndarray]]:
    """For each sample, walk its stored path root → stop-node and report the
    residual at every node along the way.

    Sanity-check helper. With ``adaptive=True`` routing, residuals should be
    monotonically nonincreasing along each path *by construction* (under the
    chosen criterion). Use ``center_only=True`` to evaluate ``‖x − c_N‖²``,
    or ``False`` for ``‖(I − ΦᵀΦ)(x − c_N)‖²``.
    """
    sample_idxs = np.asarray(sample_idxs)
    out: list[tuple[list[tuple[int, int]], np.ndarray]] = []
    for i in sample_idxs:
        x = X[i]
        path = paths[i]
        residuals = np.empty(len(path), dtype=np.float64)
        for k, nid in enumerate(path):
            diff = x - gmra.center[nid]
            Phi = gmra.basis[nid]
            if not center_only and Phi.shape[0] > 0:
                diff = diff - Phi.T @ (Phi @ diff)
            residuals[k] = float(diff @ diff)
        out.append((path, residuals))
    return out


def _chunked_residuals(
    X_sub: np.ndarray,
    c: np.ndarray,
    Phi: np.ndarray,
    center_only: bool,
    chunk: int,
) -> np.ndarray:
    n = X_sub.shape[0]
    out = np.empty(n, dtype=np.float64)
    use_projection = (not center_only) and Phi.shape[0] > 0
    for i in range(0, n, chunk):
        diffs = X_sub[i : i + chunk] - c
        if use_projection:
            diffs = diffs - diffs @ Phi.T @ Phi
        out[i : i + chunk] = np.einsum("ij,ij->i", diffs, diffs)
    return out


class GMRATokenInterp:
    """SAE-style max-activating-examples for fitted GMRA models.

    For a GMRA node, surface the K samples this node fits best (smallest
    residual against the local affine subspace ``c_N + span(Φ_N)``) and
    decode their surrounding tokens via a HuggingFace tokenizer.

    The "max activating" criterion is the same residual as
    :func:`best_fit_per_node` and the paper's
    ``E_j(x) = ‖x − P_{M_j}(x)‖`` (Allard/Chen/Maggioni, eqs 2.9 + 2.23) —
    single BLAS step against the node's local model, restricted to samples
    whose adaptive path passes through the node.
    """

    REQUIRED_COLS = ("seq_idx", "token_idx", "token_id")

    def __init__(
        self,
        gmra: GMRA,
        X: np.ndarray,
        paths: list[list[tuple[int, int]]],
        df: "pd.DataFrame",
        model_name: str,
    ):
        import pandas as pd
        from transformers import AutoTokenizer

        if X.shape[0] != len(paths):
            raise ValueError(f"X.shape[0]={X.shape[0]} != len(paths)={len(paths)}")
        if X.shape[0] != len(df):
            raise ValueError(f"X.shape[0]={X.shape[0]} != len(df)={len(df)}")
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"df is missing required columns: {missing}. Have: {list(df.columns)}"
            )

        self.gmra = gmra
        self.X = X
        self.paths = paths
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.df = df.reset_index(drop=True)

        # Per-sequence sub-DataFrames sorted by token_idx — for fast
        # context-window lookups during max_activating_examples.
        self._seq_groups: dict[int, "pd.DataFrame"] = {
            int(seq): g.sort_values("token_idx").reset_index(drop=True)
            for seq, g in self.df.groupby("seq_idx")
        }

        # Inverse routing: node_id -> indices of samples passing through.
        acc: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, p in enumerate(paths):
            for nid in p:
                acc[nid].append(i)
        self._routed: dict[tuple[int, int], np.ndarray] = {
            nid: np.array(idxs, dtype=np.int64) for nid, idxs in acc.items()
        }

    def max_activating_examples(
        self,
        node_id: tuple[int, int],
        k: int,
        context_size: int = 5,
        center_only: bool = False,
    ) -> "pd.DataFrame":
        """Top-k samples that fit ``node_id`` best, each with decoded context.

        Returns a DataFrame with columns
        ``sample_idx, seq_idx, token_idx, token_id, residual,
        focus_token, context_text, context_token_ids``,
        sorted by residual ascending. Unrouted nodes return an empty
        DataFrame.

        ``context_size`` is the number of tokens on each side of the focus
        token: a non-edge sample yields ``2*context_size + 1`` tokens.
        """
        import pandas as pd

        cols = [
            "sample_idx",
            "seq_idx",
            "token_idx",
            "token_id",
            "residual",
            "focus_token",
            "context_text",
            "context_token_ids",
        ]

        sample_idxs = self._routed.get(node_id)
        if sample_idxs is None or sample_idxs.size == 0:
            return pd.DataFrame({c: [] for c in cols})

        Phi = self.gmra.basis[node_id]
        c = self.gmra.center[node_id]
        residuals = _chunked_residuals(
            X_sub=self.X[sample_idxs],
            c=c,
            Phi=Phi,
            center_only=center_only,
            chunk=10_000,
        )

        k_eff = min(k, len(residuals))
        top = np.argpartition(residuals, k_eff - 1)[:k_eff]
        top = top[np.argsort(residuals[top])]
        top_sample_idxs = sample_idxs[top]
        top_residuals = residuals[top]

        rows = []
        for sample_idx, residual in zip(top_sample_idxs, top_residuals):
            row = self.df.iloc[int(sample_idx)]
            seq = int(row["seq_idx"])
            tok = int(row["token_idx"])
            tid = int(row["token_id"])

            window_tok, window_tid = self._fetch_window(seq, tok, context_size)

            prev_ids = [int(t) for t in window_tid[window_tok < tok]]
            next_ids = [int(t) for t in window_tid[window_tok > tok]]

            prev_text = self.tokenizer.decode(prev_ids) if prev_ids else ""
            next_text = self.tokenizer.decode(next_ids) if next_ids else ""
            focus_text = self.tokenizer.decode([tid])

            # ANSI red around the focus token. Renders coloured in terminals
            # and in Jupyter cell output; falls back to literal escape codes
            # in plain text contexts.
            context_text = f"{prev_text}\033[31m{focus_text}\033[0m{next_text}"
            context_token_ids = prev_ids + [tid] + next_ids

            rows.append(
                {
                    "sample_idx": int(sample_idx),
                    "seq_idx": seq,
                    "token_idx": tok,
                    "token_id": tid,
                    "residual": float(residual),
                    "focus_token": focus_text,
                    "context_text": context_text,
                    "context_token_ids": context_token_ids,
                }
            )

        return pd.DataFrame(rows, columns=cols)

    def _fetch_window(
        self, seq: int, tok: int, cs: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (token_idx_array, token_id_array) for tokens of ``seq``
        whose token_idx lies in [tok-cs, tok+cs] inclusive. Sorted by
        token_idx. Off-the-edge requests yield a shorter window."""
        g = self._seq_groups.get(seq)
        if g is None:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        toks = g["token_idx"].to_numpy()
        lo = int(np.searchsorted(toks, tok - cs, side="left"))
        hi = int(np.searchsorted(toks, tok + cs, side="right"))
        return toks[lo:hi], g["token_id"].to_numpy()[lo:hi]
