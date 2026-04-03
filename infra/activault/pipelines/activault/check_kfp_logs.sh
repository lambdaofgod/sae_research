#!/bin/bash
# Usage: ./check_kfp_logs.sh <pipeline-prefix> [tail-lines]
# Example: ./check_kfp_logs.sh activault-collect 100

PREFIX="${1:?Usage: $0 <pipeline-prefix> [tail-lines]}"
TAIL="${2:-0}"
NS="kubeflow"

WORKFLOW=$(minikube kubectl -- get workflows -n "$NS" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null | grep "$PREFIX")

if [ -z "$WORKFLOW" ]; then
    echo "No workflow matching '$PREFIX' found in namespace $NS"
    exit 1
fi

echo "Workflow: $WORKFLOW"

POD=$(minikube kubectl -- get pods -n "$NS" -l "workflows.argoproj.io/workflow=$WORKFLOW" -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | grep 'impl')

if [ -z "$POD" ]; then
    echo "No main container pod found yet (workflow may still be initializing)"
    minikube kubectl -- get pods -n "$NS" -l "workflows.argoproj.io/workflow=$WORKFLOW"
    exit 1
fi

echo "Pod: $POD"
echo "---"

if [ "$TAIL" -gt 0 ]; then
    minikube kubectl -- logs -n "$NS" "$POD" -c main --tail="$TAIL"
else
    minikube kubectl -- logs -n "$NS" "$POD" -c main
fi
