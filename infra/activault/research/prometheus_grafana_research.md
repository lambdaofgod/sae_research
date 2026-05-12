# Monitoring Garage S3 storage with Prometheus and Grafana on Kubernetes

**Garage exposes ~40 Prometheus metrics on its admin port at `/metrics` (port 3903), covering S3 API request rates, latencies, cluster health, disk usage, block I/O, and inter-node RPC — but critically lacks per-bucket or per-prefix breakdowns.** This means monitoring an ML activation storage pipeline requires a two-layer approach: Garage's built-in metrics for infrastructure health, plus a custom exporter that reads S3 object metadata for semantic, per-model/per-layer metrics. The Garage Helm chart natively supports ServiceMonitor CRDs, and kube-prometheus-stack can discover them across namespaces with a few key configuration values. An official Grafana dashboard exists in the Garage repository, though it is not published on grafana.com.

## Garage exposes rich infrastructure metrics on port 3903

Garage serves Prometheus-format metrics at **`http://<node>:3903/metrics`** on the admin API port. No separate metrics port exists — the admin API and metrics endpoint share port **3903**. To enable it, the `[admin]` section must be present in `garage.toml`:

```toml
[admin]
api_bind_addr = "[::]:3903"
metrics_token = "your-secret-token"   # optional; if unset, /metrics is open
```

If `metrics_token` is set, Prometheus must send `Authorization: Bearer <token>` with each scrape. Since Garage v2.0, scoped admin tokens can be created via `garage admin-token create --scope Metrics`. The environment variable `GARAGE_METRICS_TOKEN` overrides the config file value.

The complete set of metrics, verified against the official reference documentation, falls into seven categories:

**System metrics** include `garage_build_info{version="..."}`, `garage_replication_factor`, `garage_local_disk_avail{volume="data|metadata"}`, and `garage_local_disk_total{volume="data|metadata"}`. These report per-node disk capacity and usage for both the data and metadata volumes.

**Cluster health metrics** provide operational visibility: `cluster_healthy` (1 if all storage nodes connected), `cluster_available` (1 if quorum met for all partitions), `cluster_connected_nodes`, `cluster_known_nodes`, `cluster_storage_nodes`, `cluster_storage_nodes_ok`, `cluster_partitions` (always 256), `cluster_partitions_all_ok`, and `cluster_partitions_quorum`. Per-node connection status is exposed via `cluster_layout_node_connected{id, role_capacity, role_gateway, role_zone}` and `cluster_layout_node_disconnected_time`.

**S3 API metrics** are the most operationally important: `api_s3_request_counter{api_endpoint}` counts requests per S3 operation (PutObject, GetObject, ListObjectsV2, CreateMultipartUpload, etc.), `api_s3_error_counter{api_endpoint, status_code}` counts errors with HTTP codes, and `api_s3_request_duration` provides full histogram data (bucket/sum/count) per endpoint. Equivalent metrics exist for the admin API (`api_admin_*`), K2V API (`api_k2v_*`), and web endpoint (`web_request_counter`, `web_request_duration`, `web_error_counter`).

**Block manager metrics** track the data layer: `block_bytes_read` and `block_bytes_written` (counters), `block_read_duration` and `block_write_duration` (histograms), `block_ram_buffer_free_kb` (backpressure indicator), `block_delete_counter`, `block_resync_counter`, `block_resync_duration`, `block_resync_queue_length`, and **`block_resync_errored_blocks`** — the last of which should always be zero in a healthy cluster and is a critical alert target.

**RPC metrics** monitor inter-node communication: `rpc_netapp_request_counter`, `rpc_netapp_error_counter`, `rpc_timeout_counter` (should be near-zero), and `rpc_duration` histograms, all labeled with `from`, `to`, and `rpc_endpoint`.

**Metadata table metrics** cover internal database operations: `table_gc_todo_queue_length`, `table_get_request_counter`/`duration`, `table_put_request_counter`/`duration`, `table_internal_delete_counter`, `table_internal_update_counter`, `table_merkle_updater_todo_queue_length`, `table_sync_items_received`, and `table_sync_items_sent`, all labeled by `table_name` (bucket_alias, bucket_v2, block_ref, etc.).

## Garage Helm chart ServiceMonitor wires up automatically

The official Garage Helm chart (in the Garage repository at `script/helm/garage/`) and the community chart at `datahub-local/garage-helm` both support Prometheus Operator integration through two key values:

```yaml
monitoring:
  metrics:
    enabled: false          # creates a dedicated Kubernetes Service exposing port 3903
    serviceMonitor:
      enabled: false        # creates a ServiceMonitor CRD
      path: /metrics
      labels: {}
      interval: 15s
      scheme: http
      tlsConfig: {}
      scrapeTimeout: 10s
      relabelings: []
  tracing:
    sink: ""               # OpenTelemetry collector endpoint
```

Setting **`monitoring.metrics.enabled: true`** creates a separate Kubernetes Service (distinct from the S3 API service) that exposes the admin port (3903) with `prometheus.io/scrape: "true"` annotations. This separation prevents Prometheus scrape annotations from interfering with service mesh annotations on the main S3 service.

Setting **`monitoring.metrics.serviceMonitor.enabled: true`** creates a `ServiceMonitor` CRD that the Prometheus Operator will discover and use to configure Prometheus scrape targets automatically. The ServiceMonitor points at the metrics Service on path `/metrics`, port 3903, with the configured scrape interval and timeout. If your Prometheus instance requires specific labels on ServiceMonitors (the kube-prometheus-stack default behavior), add them under `monitoring.metrics.serviceMonitor.labels`.

The `datahub-local/garage-helm` community chart additionally bundles a pre-built Grafana dashboard ConfigMap, providing out-of-the-box dashboard provisioning when Grafana's sidecar is configured to discover dashboards.

For the metrics token, set it via the chart's environment values:

```yaml
environment:
  GARAGE_METRICS_TOKEN: "your-secret-metrics-token"
```

## Deploying kube-prometheus-stack alongside activault

The `prometheus-community/kube-prometheus-stack` Helm chart bundles Prometheus, Grafana, Alertmanager, node-exporter, kube-state-metrics, and the Prometheus Operator with its CRDs in a single installation. Deploy it in a dedicated `monitoring` namespace:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
kubectl create namespace monitoring
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  -n monitoring -f prometheus-values.yaml
```

The single most important configuration issue is **cross-namespace ServiceMonitor discovery**. By default, Prometheus only discovers ServiceMonitors labeled with `release: <helm-release-name>` in its own namespace. For monitoring Garage and other services in separate namespaces, these values are essential:

```yaml
prometheus:
  prometheusSpec:
    # Remove the Helm release label requirement — discover ALL ServiceMonitors
    serviceMonitorSelectorNilUsesHelmValues: false
    podMonitorSelectorNilUsesHelmValues: false
    ruleSelectorNilUsesHelmValues: false

    # Watch ALL namespaces for ServiceMonitors (empty = all)
    serviceMonitorSelector: {}
    serviceMonitorNamespaceSelector: {}
    podMonitorSelector: {}
    podMonitorNamespaceSelector: {}

    retention: 15d
    storageSpec:
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 50Gi

grafana:
  enabled: true
  adminPassword: "changeme"        # default is "prom-operator"
  sidecar:
    dashboards:
      enabled: true
      label: grafana_dashboard
      labelValue: "1"
      searchNamespace: ALL         # discover dashboard ConfigMaps in all namespaces
    datasources:
      enabled: true
  persistence:
    enabled: true
    size: 10Gi

alertmanager:
  enabled: true
```

Access Grafana via port-forward during development: `kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80`, then open `http://localhost:3000` with credentials **admin / changeme** (or whatever `adminPassword` was set to; the default if unset is `prom-operator`).

To provision the Garage Grafana dashboard, download the official JSON from the Garage repository at `script/telemetry/grafana-garage-dashboard-prometheus.json` and create a ConfigMap:

```bash
kubectl create configmap garage-grafana-dashboard -n monitoring \
  --from-file=garage.json=./grafana-garage-dashboard-prometheus.json
kubectl label configmap garage-grafana-dashboard -n monitoring grafana_dashboard="1"
```

The Grafana sidecar will automatically detect the labeled ConfigMap and load the dashboard. The kube-prometheus-stack ships with **20+ built-in Kubernetes dashboards** covering API server, nodes, pods, kubelet, statefulsets, and persistent volumes.

If the Garage chart's built-in ServiceMonitor is not used, an equivalent can be added directly in the kube-prometheus-stack values:

```yaml
prometheus:
  additionalServiceMonitors:
    - name: garage-monitor
      selector:
        matchLabels:
          app: garage
      namespaceSelector:
        matchNames:
          - garage    # or wherever Garage is deployed
      endpoints:
        - port: admin  # must match the Service's named port
          path: /metrics
          interval: 30s
```

## Existing dashboards and why a custom exporter is needed

An **official Grafana dashboard** exists in the Garage git repository at `script/telemetry/grafana-garage-dashboard-prometheus.json`. It includes panels for S3 API call rates by endpoint, block read/write throughput, disk utilization, cluster health, and RPC latencies. **No Garage dashboard is published on grafana.com** — the only source is the repository file. The community `datahub-local/garage-helm` chart bundles a dashboard ConfigMap that is auto-provisioned by Grafana's sidecar.

The fundamental gap for activation storage monitoring is that **Garage does not expose per-bucket, per-prefix, or per-object metrics**. All S3 API counters are cluster-wide, labeled only by endpoint type (PutObject, GetObject, etc.) — not by bucket or key prefix. This means tracking bytes per model, objects per hook, or tokens per run is impossible from Garage's built-in metrics alone.

A **custom Python exporter** is needed that periodically lists objects in the activault S3 bucket, parses `cfg.json` metadata files, and exposes semantic metrics. The recommended metric set:

```python
# Gauges (current state, updated each scrape cycle)
activault_storage_total_bytes                              # total bytes in bucket
activault_bytes_per_model{model}                           # bytes by model name
activault_bytes_per_layer{model, layer}                    # bytes by model + layer
activault_objects_per_hook{model, hook}                    # object counts per hook
activault_run_tokens_processed{model, run_id}              # tokens from cfg.json
activault_run_duration_seconds{model, run_id}              # duration from cfg.json
activault_run_status{model, run_id, status}                # success/failure

# Counters (instrumented in pipeline code, pushed via Pushgateway)
activault_upload_bytes_total{model, layer}                 # bytes uploaded
activault_s3_operation_latency_seconds{operation, status}  # client-side S3 latency (histogram)
```

The exporter architecture has two prongs. A **polling exporter** runs as a Deployment (port 9400), periodically listing the S3 bucket and parsing cfg.json files to compute storage gauges. A **push-based approach** instruments the pipeline code itself with `prometheus_client` counters and histograms for real-time upload throughput and S3 operation latency, pushing to a Prometheus Pushgateway after each run.

Deploy the exporter with a Service and ServiceMonitor:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: activault-exporter
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: exporter
        image: your-registry/activault-exporter:latest
        ports:
        - containerPort: 9400
          name: metrics
        env:
        - name: S3_ENDPOINT
          value: "http://garage.garage.svc:3901"
        - name: S3_BUCKET
          value: "activault"
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: activault-exporter
spec:
  selector:
    matchLabels:
      app: activault-exporter
  endpoints:
  - port: metrics
    interval: 60s
    path: /metrics
```

**Cardinality warning**: avoid putting `run_id` on high-cardinality Gauges that never get cleared. Use run_id only for the most recent N runs, or implement a TTL clearing mechanism. Use the `model` label freely but treat `run_id` as ephemeral metadata.

The existing open-source `ribbybibby/s3_exporter` can supplement this by providing `s3_objects_size_sum`, `s3_objects_total`, and `s3_list_duration_seconds` metrics with prefix-based probing — useful as a quick start before building a full custom exporter. It supports custom S3 endpoints (`--s3.endpoint-url`) and works with Garage.

## KFP and Argo expose pipeline metrics natively

Kubeflow Pipelines components expose Prometheus metrics, though not always with default scrape annotations. The **KFP API Server** (`ml-pipeline`) serves `/metrics` on port **8888** with counters for pipeline, experiment, job, and run server operations plus Go runtime metrics. You may need to create a ServiceMonitor manually since the service historically lacks `prometheus.io/scrape` annotations.

The **Argo Workflow Controller** (the execution engine underneath KFP) provides the richest pipeline metrics at `:9090/metrics`:

- `argo_workflows_count{status}` — gauge of workflows by phase (Pending, Running, Succeeded, Failed)
- `argo_workflows_pods_count{status}` — workflow pod counts by status
- `argo_workflows_operation_duration_seconds` — controller operation duration histogram
- `argo_workflows_queue_depth_count`, `argo_workflows_queue_latency` — work queue health
- `argo_workflows_error_count{cause}` — errors by cause
- `argo_workflows_cronworkflows_triggered_total` — scheduled workflow triggers

Argo also supports **custom user-defined metrics in workflow YAML**, which is the most idiomatic way to expose per-pipeline metrics:

```yaml
spec:
  metrics:
    prometheus:
    - name: activation_collection_duration
      help: "Duration of activation collection"
      gauge:
        value: "{{workflow.duration}}"
      labels:
      - key: model
        value: "{{workflow.parameters.model_name}}"
```

An existing Grafana dashboard for Argo Workflows is published on grafana.com: **Dashboard ID 21393** (for Argo v3.6+) or **20348** (for v3.5 and below).

## Recommended dashboard panels and PromQL queries

For a complete monitoring setup, build three Grafana dashboards covering infrastructure, application, and pipeline layers.

**Garage infrastructure dashboard** (import the official JSON, then customize): Use `1 - (garage_local_disk_avail{volume="data"} / garage_local_disk_total{volume="data"})` for disk utilization percentage. Track S3 throughput with `rate(api_s3_request_counter[5m])` by `api_endpoint`. Alert on `block_resync_errored_blocks > 0` and `cluster_healthy == 0`. Monitor S3 latency p99 with `histogram_quantile(0.99, rate(api_s3_request_duration_bucket[5m]))`.

**Activault storage dashboard** (custom): Total storage as a Stat panel (`activault_storage_total_bytes`). Storage by model as a stacked time series (`activault_bytes_per_model`). Object counts by hook as a bar chart. Daily growth rate via `delta(activault_storage_total_bytes[24h])`. **30-day storage projection** using `predict_linear(activault_storage_total_bytes[7d], 30*24*3600)` — critical for capacity planning. Upload throughput as `rate(activault_upload_bytes_total[5m])`. Client-side S3 latency heatmap from `rate(activault_s3_op_latency_seconds_bucket[5m])`.

**Pipeline runs dashboard** (Argo + KFP): Workflow status counts from `argo_workflows_count{status}`. Error rate via `rate(argo_workflows_error_count[5m])`. Custom per-model collection duration from Argo custom metrics. Queue depth for controller health.

## Conclusion

The monitoring architecture for this project breaks into three clean layers. Garage provides solid infrastructure observability out of the box — the `/metrics` endpoint on port 3903 covers everything from S3 API latencies to cluster partition health, and the Helm chart's `monitoring.metrics.serviceMonitor.enabled` wires it directly into Prometheus Operator. The kube-prometheus-stack deploys the full monitoring stack in one Helm install, but requires setting `serviceMonitorSelectorNilUsesHelmValues: false` and `serviceMonitorNamespaceSelector: {}` to discover ServiceMonitors across namespaces — this is the most common pitfall. The critical architectural insight is that **Garage's metrics are entirely infrastructure-level**: they cannot tell you how much storage model X uses or how many activations hook Y has collected. A custom exporter that reads S3 bucket contents and parses cfg.json metadata is not optional — it is the only path to the semantic, per-model/per-layer metrics that make this monitoring stack genuinely useful for ML activation storage.
