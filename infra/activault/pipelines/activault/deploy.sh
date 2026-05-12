#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHART_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$CHART_DIR")"

VARIANT="${1:-flashattn}"

case "$VARIANT" in
  flashattn)
    IMAGE="activault-flashattn:latest"
    DOCKERFILE="$SCRIPT_DIR/Dockerfile.flashattn"
    JOB_MANIFEST="$SCRIPT_DIR/collector-job-flashattn.yaml"
    JOB_NAME="activault-collector-flashattn"
    ;;
  devel)
    IMAGE="activault-devel:latest"
    DOCKERFILE="$SCRIPT_DIR/Dockerfile"
    JOB_MANIFEST="$SCRIPT_DIR/collector-job.yaml"
    JOB_NAME="activault-collector"
    ;;
  *)
    echo "Usage: $0 [flashattn|devel]"
    exit 1
    ;;
esac

kubectl() { minikube kubectl -- "$@"; }

echo "=== Building $IMAGE ==="
eval $(minikube docker-env)
docker build -t "$IMAGE" -f "$DOCKERFILE" "$SCRIPT_DIR"

echo "=== Updating chart dependencies ==="
helm dependency update "$CHART_DIR"

echo "=== Installing/upgrading chart ==="
helm upgrade --install activault "$CHART_DIR"

echo "=== Creating HF token secret ==="
kubectl delete secret hf-token-secret --ignore-not-found
kubectl create secret generic hf-token-secret --from-file=token="$HOME/.keys/hf_token.txt"

echo "=== Deploying job: $JOB_NAME ==="
kubectl delete job "$JOB_NAME" --ignore-not-found
kubectl apply -f "$JOB_MANIFEST"

echo "=== Waiting for pod to start ==="
for i in $(seq 1 30); do
  PHASE=$(kubectl get pods -l "job-name=$JOB_NAME" -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "Pending")
  echo "  pod phase: $PHASE"
  if [[ "$PHASE" != "Pending" ]]; then break; fi
  sleep 2
done

echo "=== Logs (last 50 lines) ==="
kubectl logs -l "job-name=$JOB_NAME" --tail=50 2>&1 || echo "(no logs yet)"
