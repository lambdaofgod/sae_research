"""Activault-based training smoke test.

Drives ~run_sae_training~ down its Activault-path branch (which
constructs ~ResilientS3RCache~ + ~ActivaultS3ActivationBuffer~) so
the cache shutdown path that hung PID 783351 for 5 days is actually
exercised. Without this test, ~smoke_test_training.py~'s stub buffer
bypasses the cache entirely and a regressed shutdown fix would
silently pass.

The clean-shutdown signal is: after ~buffer.close()~ →
~cache.finalize()~ → ~_stop_downloading~ (the patched path), no
~multiprocessing~ children should survive. That's the only assertion
we make about teardown — checking GPU memory delta is too noisy
(PyTorch's caching allocator keeps blocks reserved even after
~empty_cache()~).

Usage:
    bash scripts/with-k8s-env.sh \\
        uv run python smoke_tests/smoke_test_activault_training.py [--cache=<name>]

If ~--cache~ is omitted, the alphabetically-first cache is used and
the alternatives are logged so you can re-run with a different one.

When run *without* ~with-k8s-env.sh~, the boto3 calls fail with
~NoCredentialsError~ / ~EndpointConnectionError~ / ~ClientError~;
those are caught at the top level and re-raised with a hint pointing
at the helper script.
"""

import logging
import multiprocessing
import os
import shutil
import tempfile
import time

import boto3
import fire
import torch as t
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
)

from sae_research.training.config import (
    ActivaultConfig,
    build_trainer_config,
    resolve_architecture,
)
from sae_research.training.train import trainSAE
from sae_research.training.utils import create_activault_buffer


logger = logging.getLogger(__name__)

S3_BUCKET = "activations"
SETUP_HINT = (
    "Couldn't reach activault S3.\n"
    "This usually means K8s port-forwards / S3 credentials aren't set.\n"
    "Re-run inside the helper:\n"
    "    bash scripts/with-k8s-env.sh \\\n"
    "        uv run python smoke_tests/smoke_test_activault_training.py [...]"
)


def _list_activault_caches(s3_client, bucket: str) -> list[str]:
    """Return all cache prefixes (those with a metadata.json file)."""
    paginator = s3_client.get_paginator("list_objects_v2")
    prefixes: list[str] = []
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/metadata.json"):
                prefixes.append(key[: -len("/metadata.json")])
    return prefixes


def _pick_cache(requested: str | None, available: list[str]) -> str:
    if not available:
        raise SystemExit(
            f"No activault caches found in s3://{S3_BUCKET}/. "
            "Either no datasets have been precomputed yet, or the bucket "
            "is empty. If you expected caches to be present, see "
            "scripts/with-k8s-env.sh."
        )
    available = sorted(available)
    if requested is not None:
        if requested not in available:
            raise SystemExit(
                f"Requested cache {requested!r} not found in "
                f"s3://{S3_BUCKET}/. Available: {available}"
            )
        return requested
    chosen = available[0]
    others = available[1:]
    print(f"Picked cache (alphabetically first): {chosen}")
    if others:
        print(f"  Other available caches: {others}")
        print("  Re-run with --cache=<name> to pick a different one.")
    return chosen


def _orphan_children(baseline_pids: set[int | None]) -> list:
    return [p for p in multiprocessing.active_children() if p.pid not in baseline_pids]


def main(cache: str | None = None):
    """Run a tiny activault-driven training pass and verify clean
    shutdown of ResilientS3RCache.

    Args:
        cache: optional S3 prefix to use. If omitted, picks the
            alphabetically-first cache available and logs the
            alternatives.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    s3_client = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))

    try:
        available = _list_activault_caches(s3_client, S3_BUCKET)
    except (NoCredentialsError, EndpointConnectionError, ClientError) as e:
        raise SystemExit(f"{SETUP_HINT}\n\nOriginal error: {e}")

    chosen = _pick_cache(cache, available)

    # Baseline state for clean-shutdown assert
    baseline_pids: set[int | None] = {p.pid for p in multiprocessing.active_children()}

    activault = ActivaultConfig(
        s3_bucket=S3_BUCKET,
        s3_prefix=chosen,
        s3_buffer_size=2,
        s3_workers=2,
    )

    save_dir = tempfile.mkdtemp(prefix="activault_smoke_")
    device = "cuda" if t.cuda.is_available() else "cpu"
    sae_batch_size = 64
    buffer = None

    try:
        try:
            buffer, metadata = create_activault_buffer(
                activault, sae_batch_size=sae_batch_size, device=device
            )
        except (NoCredentialsError, EndpointConnectionError, ClientError) as e:
            raise SystemExit(f"{SETUP_HINT}\n\nOriginal error: {e}")

        activation_dim = metadata["shape"][-1]
        print(
            f"Buffer ready. activation_dim={activation_dim}, "
            f"dtype={metadata['dtype']}, prefix={chosen}"
        )

        # Trainer config sized off the cache's activation_dim. Skip
        # k-annealing entirely (k_anneal_steps=None) — the annealing
        # schedule starts at activation_dim and would require
        # dict_size >= activation_dim. We only run 5 steps; annealing
        # makes no sense at this scale.
        trainer_path, dict_class_path = resolve_architecture("batch_top_k")
        trainer_config = build_trainer_config(
            trainer=trainer_path,
            dict_class=dict_class_path,
            activation_dim=activation_dim,
            dict_size=max(2 * activation_dim, 1024),
            seed=0,
            lr=1e-4,
            steps=10,
            device=device,
            layer=0,
            model_name="activault_smoke_test",
            submodule_name="resid",
            warmup_steps=1,
            decay_start_fraction=0.8,
            k=8,
            k_anneal_steps=None,
        )

        print("\n=== Training (5 steps) ===")
        trainSAE(
            data=buffer,
            trainer_configs=[trainer_config],
            steps=5,
            save_dir=save_dir,
            log_steps=1,
            normalize_activations=False,
            device=device,
        )
        print("Training done.")

    finally:
        # Explicitly close the buffer → cache.finalize() →
        # ResilientS3RCache._stop_downloading. This is the path the
        # patch is meant to fix.
        if buffer is not None:
            print("\n=== Closing buffer (exercises shutdown path) ===")
            shutdown_start = time.monotonic()
            buffer.close()
            shutdown_elapsed = time.monotonic() - shutdown_start
            print(f"  buffer.close() returned in {shutdown_elapsed:.2f}s")
        shutil.rmtree(save_dir, ignore_errors=True)

    # Give workers a beat to fully reap before we check
    time.sleep(1)

    print("\n=== Verifying clean shutdown ===")
    leftover = _orphan_children(baseline_pids)
    if leftover:
        # Force kill so we don't leak — this is a smoke test, not a
        # production cleanup path; print first then kill.
        leftover_pids = [p.pid for p in leftover]
        for p in leftover:
            p.kill()
        raise SystemExit(
            f"FAIL: {len(leftover)} multiprocessing child(ren) still alive "
            f"after buffer.close(): pids={leftover_pids}. The "
            f"ResilientS3RCache shutdown patch is not working."
        )
    print(f"PASS: no orphan multiprocessing children (baseline: {len(baseline_pids)})")

    print("\nActivault training smoke test passed!")


if __name__ == "__main__":
    fire.Fire(main)
