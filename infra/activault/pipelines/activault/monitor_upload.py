#!/usr/bin/env python3
"""Monitor Garage bucket upload progress and estimate completion time.

Usage:
    ./monitor_upload.py [--watch SECONDS] [--total-gb GB]

Examples:
    ./monitor_upload.py                          # 60s watch, 1012.5 GB total
    ./monitor_upload.py --watch 120 --total-gb 500

The Garage admin token is read from the cluster secret automatically.
"""

import argparse
import json
import subprocess
import sys
import time


BUCKET_ID = "41e65f26651292fde480676afc5e934bf1ea7524d733a467739d53ad03f566f9"
ADMIN_URL = "http://activault-garage-0.activault-garage-headless:3903/v2/GetBucketInfo?id={bucket_id}"


def get_admin_token():
    result = subprocess.run(
        [
            "minikube",
            "kubectl",
            "--",
            "get",
            "secret",
            "activault-garage-rpc-secret",
            "-o",
            "jsonpath={.data.adminToken}",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        sys.exit(f"Failed to read admin token from cluster: {result.stderr.strip()}")
    import base64

    return base64.b64decode(result.stdout).decode()


def get_bucket_info(token):
    url = ADMIN_URL.format(bucket_id=BUCKET_ID)
    result = subprocess.run(
        [
            "minikube",
            "kubectl",
            "--",
            "run",
            "curl-monitor",
            "--rm",
            "-it",
            "--image=curlimages/curl",
            "--restart=Never",
            "--",
            "-s",
            "-H",
            f"Authorization: Bearer {token}",
            url,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # strip trailing pod deletion message
    raw = result.stdout.strip()
    json_end = raw.rfind("}") + 1
    return json.loads(raw[:json_end])


def fmt_bytes(b):
    if b >= 1e12:
        return f"{b / 1e12:.2f} TB"
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.1f} MB"
    return f"{b} B"


def fmt_duration(seconds):
    if seconds < 0:
        return "N/A"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"


def main():
    parser = argparse.ArgumentParser(description="Monitor Garage upload progress")
    parser.add_argument(
        "--watch",
        type=int,
        default=60,
        help="Observation window in seconds (default: 60)",
    )
    parser.add_argument(
        "--total-gb",
        type=float,
        default=1012.5,
        help="Expected total size in GB (default: 1012.5)",
    )
    args = parser.parse_args()

    total_bytes = args.total_gb * 1e9
    token = get_admin_token()

    print(f"Sampling start...")
    start_info = get_bucket_info(token)
    start_bytes = start_info["bytes"]
    start_objects = start_info["objects"]
    print(f"  Objects: {start_objects}  Size: {fmt_bytes(start_bytes)}")

    print(f"Waiting {args.watch}s...")
    time.sleep(args.watch)

    print(f"Sampling end...")
    end_info = get_bucket_info(token)
    end_bytes = end_info["bytes"]
    end_objects = end_info["objects"]
    print(f"  Objects: {end_objects}  Size: {fmt_bytes(end_bytes)}")

    delta_bytes = end_bytes - start_bytes
    delta_objects = end_objects - start_objects
    speed_bps = delta_bytes / args.watch if args.watch > 0 else 0

    print()
    print(f"--- Results ({args.watch}s window) ---")
    print(f"New objects:  {delta_objects}")
    print(f"Data written: {fmt_bytes(delta_bytes)}")
    print(f"Upload speed: {fmt_bytes(speed_bps)}/s ({fmt_bytes(speed_bps * 3600)}/h)")

    pct = (end_bytes / total_bytes) * 100
    remaining = total_bytes - end_bytes
    print()
    print(
        f"Progress:     {fmt_bytes(end_bytes)} / {fmt_bytes(total_bytes)} ({pct:.1f}%)"
    )

    if speed_bps > 0:
        eta_seconds = remaining / speed_bps
        print(f"ETA:          {fmt_duration(eta_seconds)}")
    else:
        print("ETA:          N/A (no data written during observation)")


if __name__ == "__main__":
    main()
