#!/usr/bin/env bash
# Source K8s service credentials into env vars, ensure port-forwards
# are running, then exec the given command.
#
# Usage:
#   bash smoke_tests/with-k8s-env.sh uv run python smoke_tests/smoke_test_training.py
set -euo pipefail

# ── Use minikube kubectl if kubectl is not available ───────────────────────
if ! command -v kubectl &>/dev/null; then
    kubectl() { minikube kubectl -- "$@"; }
fi

SECRET_NAME="activault-s3-creds"

# ── Read S3 creds from K8s secret directly into env vars ───────────────────
export AWS_ACCESS_KEY_ID=$(kubectl get secret "$SECRET_NAME" -o jsonpath='{.data.AWS_ACCESS_KEY_ID}' | base64 -d)
export AWS_SECRET_ACCESS_KEY=$(kubectl get secret "$SECRET_NAME" -o jsonpath='{.data.AWS_SECRET_ACCESS_KEY}' | base64 -d)
export AWS_DEFAULT_REGION=$(kubectl get secret "$SECRET_NAME" -o jsonpath='{.data.AWS_DEFAULT_REGION}' | base64 -d)

# Garage S3 endpoint — via local port-forward
export AWS_ENDPOINT_URL="http://localhost:3900"
export MLFLOW_S3_ENDPOINT_URL="http://localhost:3900"

# ── Ensure port-forwards are running ──────────────────────────────────────
ensure_port_forward() {
    local svc="$1" local_port="$2" remote_port="$3" ns="${4:-default}"
    if ! ss -tlnp 2>/dev/null | grep -q ":${local_port} "; then
        echo "Starting port-forward: ${svc} ${local_port}:${remote_port} (ns=${ns})"
        kubectl port-forward -n "${ns}" "svc/${svc}" "${local_port}:${remote_port}" &>/dev/null &
        sleep 1
    fi
}

ensure_port_forward activault-proxy 8085 8085 default    # MLflow
ensure_port_forward activault-garage 3900 3900 default   # Garage S3
ensure_port_forward ml-pipeline 8087 8888 kubeflow       # KFP API

# ── Exec the command ─────────────────────────────────────────────────────
exec "$@"
