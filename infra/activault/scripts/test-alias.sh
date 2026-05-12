#!/bin/sh
# Read admin token from cluster secret
AUTH="Bearer $(minikube kubectl -- get secret activault-garage-rpc-secret -o jsonpath='{.data.adminToken}' | base64 -d)"
ADMIN="http://activault-garage-0.activault-garage-headless:3903"
BUCKET_ID="192a658f541dc061584f93259788da68a899df83f9b224bbde310506cf51a5a7"

echo "--- format: alias.global ---"
curl -s -X POST -H "Authorization: $AUTH" -H "Content-Type: application/json" "$ADMIN/v2/AddBucketAlias" -d "{\"bucketId\":\"$BUCKET_ID\",\"alias\":{\"global\":\"activations\"}}"
echo ""

echo "--- format: globalAlias string ---"
curl -s -X POST -H "Authorization: $AUTH" -H "Content-Type: application/json" "$ADMIN/v2/AddBucketAlias" -d "{\"bucketId\":\"$BUCKET_ID\",\"globalAlias\":\"activations\"}"
echo ""

echo "--- format: UpdateBucket with globalAliases ---"
curl -s -X POST -H "Authorization: $AUTH" -H "Content-Type: application/json" "$ADMIN/v2/UpdateBucket" -d "{\"id\":\"$BUCKET_ID\",\"globalAliases\":[\"activations\"]}"
echo ""

echo "--- PutBucketGlobalAlias ---"
curl -s -X POST -H "Authorization: $AUTH" -H "Content-Type: application/json" "$ADMIN/v2/PutBucketGlobalAlias" -d "{\"id\":\"$BUCKET_ID\",\"alias\":\"activations\"}"
echo ""

echo "--- try v1 style ---"
curl -s -X PUT -H "Authorization: $AUTH" "$ADMIN/v1/bucket/alias/global?id=$BUCKET_ID&alias=activations"
echo ""
