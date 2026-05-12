"""Activault S3 storage exporter for Prometheus.

Periodically lists objects in the activations bucket, parses cfg.json
metadata, and exposes per-model/per-hook storage gauges on :9400/metrics.
"""

import json
import logging
import os
import time

import boto3
from prometheus_client import Gauge, start_http_server

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("activault-exporter")

# ── Config ────────────────────────────────────────────────────────────────────

S3_ENDPOINT = os.environ["S3_ENDPOINT"]
S3_BUCKET = os.environ.get("S3_BUCKET", "activations")
S3_ACCESS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
S3_SECRET_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
S3_REGION = os.environ.get("AWS_DEFAULT_REGION", "garage")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "300"))  # seconds
PORT = int(os.environ.get("PORT", "9400"))

# ── Metrics ───────────────────────────────────────────────────────────────────

# Bucket-level
storage_total_bytes = Gauge(
    "activault_storage_total_bytes", "Total bytes in the activations bucket"
)
storage_total_objects = Gauge(
    "activault_storage_total_objects", "Total objects in the activations bucket"
)

# Per-run numeric gauges (for PromQL: stat panels, bar charts, time series)
bytes_per_run = Gauge("activault_bytes_per_run", "Bytes per run", ["run", "model"])
run_tokens = Gauge(
    "activault_run_tokens", "Total tokens processed per run", ["run", "model"]
)
run_completeness = Gauge(
    "activault_run_completeness_ratio",
    "Collection completeness (batches_processed / n_batches)",
    ["run", "model"],
)

# Per-hook numeric gauges (for PromQL)
objects_per_hook = Gauge(
    "activault_objects_per_hook",
    "Actual object count per hook",
    ["run", "model", "hook"],
)
bytes_per_hook = Gauge(
    "activault_bytes_per_hook",
    "Bytes per hook",
    ["run", "model", "hook"],
)

# Summary gauges for dashboard tables (value=bytes, all details as labels)
# One query + labelsToFields = clean table, no joins needed
run_summary = Gauge(
    "activault_run_summary",
    "Per-run summary for dashboard table (value=storage bytes)",
    [
        "run",
        "model",
        "dtype",
        "d_model",
        "dataset",
        "batch_size",
        "seq_length",
        "tokens",
        "batches_done",
        "batches_target",
        "completeness_pct",
        "files_expected",
    ],
)
hook_summary = Gauge(
    "activault_hook_summary",
    "Per-hook summary for dashboard table (value=storage bytes)",
    ["run", "model", "hook", "objects", "files_expected", "completeness_pct"],
)

ALL_LABELED_GAUGES = [
    bytes_per_run,
    run_tokens,
    run_completeness,
    objects_per_hook,
    bytes_per_hook,
    run_summary,
    hook_summary,
]

# ── S3 client ─────────────────────────────────────────────────────────────────


def make_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=S3_REGION,
    )


# ── Collection ────────────────────────────────────────────────────────────────


def collect():
    """List bucket, parse structure, update all gauges."""
    s3 = make_client()
    log.info("Starting collection from s3://%s", S3_BUCKET)

    run_bytes: dict[str, int] = {}
    hook_bytes: dict[tuple[str, str], int] = {}
    hook_objects: dict[tuple[str, str], int] = {}
    total_bytes = 0
    total_objects = 0
    cfg_cache: dict[str, dict] = {}

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            size = obj["Size"]
            total_bytes += size
            total_objects += 1

            parts = key.split("/")
            if len(parts) < 2:
                continue

            run_name = parts[0]
            run_bytes[run_name] = run_bytes.get(run_name, 0) + size

            if parts[-1] == "cfg.json":
                try:
                    resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
                    cfg_cache[run_name] = json.loads(resp["Body"].read())
                except Exception:
                    log.warning("Failed to read %s", key, exc_info=True)
                continue

            # Shard file: <run>/<hook>/<file>.saved.pt
            if len(parts) == 3:
                hook = parts[1]
                hk = (run_name, hook)
                hook_bytes[hk] = hook_bytes.get(hk, 0) + size
                hook_objects[hk] = hook_objects.get(hk, 0) + 1

    # ── Update gauges ─────────────────────────────────────────────────────

    storage_total_bytes.set(total_bytes)
    storage_total_objects.set(total_objects)

    for g in ALL_LABELED_GAUGES:
        g._metrics.clear()

    for run_name, rbytes in run_bytes.items():
        cfg = cfg_cache.get(run_name, {})
        tc = cfg.get("transformer_config", {})
        dc = cfg.get("data_config", {})
        uc = cfg.get("upload_config", {})
        model = tc.get("model_name", "unknown")

        # Numeric gauges for PromQL
        bytes_per_run.labels(run=run_name, model=model).set(rbytes)

        tokens = cfg.get("total_tokens", 0)
        batches_done = cfg.get("batches_processed", 0)
        batches_target = dc.get("n_batches", 0)
        n_files = cfg.get("n_total_files", 0)
        n_hooks = len(uc.get("hooks", []))
        completeness = batches_done / batches_target if batches_target > 0 else 0

        if cfg:
            run_tokens.labels(run=run_name, model=model).set(tokens)
            run_completeness.labels(run=run_name, model=model).set(completeness)

            # Summary gauge for table display
            run_summary.labels(
                run=run_name,
                model=model,
                dtype=tc.get("dtype", "?"),
                d_model=str(cfg.get("d_model", "?")),
                dataset=dc.get("data_key", "?"),
                batch_size=str(dc.get("batch_size", "?")),
                seq_length=str(dc.get("seq_length", "?")),
                tokens=str(tokens),
                batches_done=str(batches_done),
                batches_target=str(batches_target),
                completeness_pct=f"{completeness:.0%}",
                files_expected=str(n_files),
            ).set(rbytes)

            # Per-hook expected and completeness
            expected_per_hook = n_files / n_hooks if n_hooks > 0 else 0
            for hook in uc.get("hooks", []):
                actual = hook_objects.get((run_name, hook), 0)
                hc = actual / expected_per_hook if expected_per_hook > 0 else 0
                hook_summary.labels(
                    run=run_name,
                    model=model,
                    hook=hook,
                    objects=str(actual),
                    files_expected=str(int(expected_per_hook)),
                    completeness_pct=f"{hc:.0%}",
                ).set(hook_bytes.get((run_name, hook), 0))

    for (run_name, hook), hbytes in hook_bytes.items():
        cfg = cfg_cache.get(run_name, {})
        model = cfg.get("transformer_config", {}).get("model_name", "unknown")
        bytes_per_hook.labels(run=run_name, model=model, hook=hook).set(hbytes)
        objects_per_hook.labels(run=run_name, model=model, hook=hook).set(
            hook_objects[(run_name, hook)]
        )

    log.info(
        "Collection done: %d objects, %.2f GB, %d runs",
        total_objects,
        total_bytes / 1e9,
        len(run_bytes),
    )


# ── Main loop ─────────────────────────────────────────────────────────────────


def poll_loop():
    while True:
        try:
            collect()
        except Exception:
            log.error("Collection failed", exc_info=True)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    log.info("Starting activault-exporter on :%d (poll every %ds)", PORT, POLL_INTERVAL)
    start_http_server(PORT)
    poll_loop()
