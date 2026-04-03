# ML platform architecture: a tools-first recommendation guide

**The strongest open-source ML platform stack for LLM + SAE research combines Kubeflow Pipelines v2 for orchestration, MLflow for experiment tracking and model registry, BentoML for model serving, and Kueue for GPU job scheduling—all on Kubernetes.** Manning's *Machine Learning Platform Engineering* (Tan Wei Hao, Padmanabhan, Mallya; February 2026) covers this exact stack across 13 chapters, providing the most current reference architecture for building an internal ML platform from scratch. This report maps every topic in the user's query to specific book chapters, tool configurations, and production-ready architecture patterns.

---

## The Manning book covers this stack end-to-end

The book's correct authorship is **Benjamin Tan Wei Hao, Shanoop Padmanabhan, and Varun Mallya**—not Gerasimov. It was published in February 2026 (ISBN 9781633437333, 504 pages) and uses three capstone projects: an OCR system (YOLOv8), a movie recommender, and a RAG-based LLM assistant (DakkaBot). The tooling stack aligns almost perfectly with the query.

**Direct chapter-to-topic mapping:**

| Topic | Primary chapters | Key content |
|-------|-----------------|-------------|
| KFP orchestration | **Ch 5** (dedicated), Ch 7–9 (applied) | KFP v2 components, DAG construction, artifact types, data passing |
| MLflow tracking + registry | **Ch 4** (core), **Ch 9** (applied) | Experiment tracking, model registry, MinIO artifact store |
| BentoML serving | **Ch 6** (intro), **Ch 10** (deep dive) | Service/Runners, packaging, MLflow integration, **KServe as alternative** |
| Feature stores | **Ch 4** (Feast + Redis) | Feature registration, retrieval, feature server, Feast UI |
| GPU scheduling | **Ch 3** (K8s foundations), **Ch 7** (affinity/tolerations) | Node affinity, tolerations for GPU node selection |
| Monorepo consolidation | *Not directly covered* | Book uses project-based repos; no monorepo chapter |
| LLM caching patterns | **Ch 13** (Section 13.4.3) | LLM cost optimization caching strategies |

Chapter 10 is particularly valuable because it compares three serving approaches side-by-side: **BentoML** (primary), **MLflow inference** (lightweight), and **KServe** (enterprise alternative). Chapter 9 demonstrates the full MLflow integration pattern inside a KFP pipeline for the movie recommender project, logging experiments, registering models, and storing artifacts in MinIO.

The book's Appendix A contains setup instructions for every component—K3s/MicroK8s, Argo CD, Kubeflow, MLflow, Redis, BentoML/Yatai, and Evidently—making it a practical bootstrap guide. The GitHub repository lives at `github.com/practical-mlops/object-detection-project`.

---

## Kubeflow ecosystem architecture for training and serving

The Kubeflow 1.11 release (December 2025) represents a major architectural shift. The **Training Operator v2** replaces framework-specific CRDs (PyTorchJob, TFJob) with a unified `TrainJob` API built on Kubernetes JobSet. Platform admins define `ClusterTrainingRuntime` resources specifying infrastructure defaults (GPU allocation, scheduling policy, failure handling), while data scientists submit lightweight `TrainJob` manifests referencing those runtimes. V2 includes **built-in LLM fine-tuning blueprints** for Llama 3.2 and Qwen 2.5 via TorchTune, supporting LoRA, QLoRA, and DoRA.

For GPU job queuing, **Kueue is the recommended integration** with Training Operator v2. The architecture works at two levels: Kueue controls *admission* (deciding when a job can start based on quota, priority, and fair-sharing rules), while the Kubernetes scheduler or Volcano handles *placement* (deciding where pods run). A concrete Kueue setup requires three resources:

- **ResourceFlavor** mapping to GPU node labels (e.g., `nvidia-tesla-a100`)
- **ClusterQueue** defining quota pools per flavor (e.g., 8 A100 GPUs with borrowing limits)
- **LocalQueue** in each team namespace, pointing to the ClusterQueue

Multiple ClusterQueues can share resources via **Cohorts** with Dominant Resource Share (DRS) fair-sharing. Jobs are annotated with a single label (`kueue.x-k8s.io/queue-name: user-queue`) to enter the scheduling system. Kueue supports `StrictFIFO` and `BestEffortFIFO` strategies, preemption policies, and topology-aware scheduling for optimal GPU placement.

For **job dependencies** (training should wait if prerequisite data isn't in S3), the Training Operator has no built-in mechanism. Three patterns work:

1. **Init containers** on the PyTorchJob/TrainJob that poll S3 using `aws s3 ls` and exit non-zero if data is missing, causing the pod to retry with `restartPolicy: OnFailure`
2. **KFP pipeline DAGs** where a data-validation component must succeed before the training component launches
3. **Trainer v2 dataset initializers** that handle S3 data fetching as part of the job lifecycle

For **hyperparameter sweeps across architectures**, Katib remains the primary tool. A Katib `Experiment` CRD defines the search space, objective metric, and algorithm (random, Bayesian optimization, TPE, Hyperband, CMAES), then each trial spawns a full PyTorchJob/TrainJob. **Katib supports parallel trials**—setting `parallelTrialCount: 3` with `maxTrialCount: 12` runs 3 concurrent GPU jobs across 12 total configurations. Multi-architecture sweeps combine Katib with Kueue ResourceFlavors or `nodeSelector` tolerations per trial template to target different GPU types.

**KServe** (now a CNCF incubating project, v0.15.2) provides a complementary serving layer with capabilities BentoML lacks: scale-to-zero, canary rollouts, InferenceGraph for multi-model pipelines (Sequence, Switch, Ensemble, Splitter routing), and ModelMesh for high-density multi-model serving. KServe's `InferenceGraph` CRD could orchestrate an LLM→SAE pipeline natively. However, BentoML offers faster iteration with its Python-first API—the two are complementary, and **BentoML can deploy *to* KServe** as a serving backend.

---

## BentoML patterns for LLM + SAE co-hosting

BentoML's post-1.2 API uses `@bentoml.service` class decorators with `@bentoml.api` method decorators, replacing the legacy Runner architecture. For hosting a base LLM with multiple SAE checkpoints hooked into its forward pass, two architecture patterns apply.

**Pattern A: Single service, multiple SAEs in memory.** Load one LLM in `__init__`, load all SAE checkpoints into a dictionary keyed by variant name, and use PyTorch's `register_forward_hook()` to intercept activations at the target layer. API endpoints accept a `sae_variant` parameter for routing. Since SAE models are typically **10–100 MB** each versus multi-GB LLMs, dozens of SAE variants can coexist in GPU memory alongside a single base model:

```python
@bentoml.service(resources={"gpu": 1}, workers=1)
class MultiSAEService:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(...)
        self.saes = {name: torch.load(f"saes/{name}.pt") for name in variants}
    
    @bentoml.api
    async def analyze(self, prompt: str, sae_variant: str = "default") -> dict:
        sae = self.saes[sae_variant]
        # Hook SAE into LLM forward pass, run inference
```

**Pattern B: Distributed services with `bentoml.depends()`.** When SAE variants need independent scaling or resource isolation, separate the base LLM into its own service and create lightweight SAE services that call it via dependency injection. Each SAE service runs in a separate container with independent autoscaling, while the shared LLM service runs once on GPU:

```python
@bentoml.service(resources={"gpu": 1})
class BaseLLMService:
    # Expensive GPU service, loaded once

@bentoml.service(resources={"cpu": "2"})
class SAEVariantA:
    llm = bentoml.depends(BaseLLMService)  # Auto-routed inter-service call
```

For **batch vs. real-time inference**, BentoML provides three mechanisms. Adaptive batching (`@bentoml.api(batchable=True, max_batch_size=32, max_latency_ms=1000)`) dynamically groups concurrent requests server-side—ideal for embedding endpoints. The `@bentoml.task` decorator (introduced in 1.3+) creates async task queues with submit/status/result endpoints for fire-and-forget long-running jobs. For massive offline processing, deploy-then-terminate patterns use `bentoml.deployment.create()` programmatically.

**Kubernetes deployment** uses the **Yatai operator**, which provides three CRDs: `BentoRequest`, `Bento`, and `BentoDeployment`. Yatai handles image building (via Kaniko), deployment lifecycle, and separate pod scaling for API servers and runners. Helm installation is straightforward via `helm install yatai bentoml/yatai -n yatai-system`. Note that **Yatai 2.0 for BentoML 1.2+ is under active development**—current stable Yatai works fully with BentoML 1.1 patterns. BentoML CRDs integrate natively with Kubeflow since version 1.7.

---

## MLflow registry patterns for SAE experiment management

MLflow's model registry provides three complementary metadata layers for SAE models. The `metadata` parameter on `log_model()` stores immutable properties in the MLmodel YAML file—ideal for fixed attributes like SAE k-values, hook points, and architecture type. Run-level `log_param()` and `log_metric()` calls are searchable through the MLflow UI and API—use these for training hyperparameters and evaluation metrics. Mutable tags via `set_tag()` and `set_model_version_tag()` handle lifecycle labels and deployment status.

A concrete SAE experiment logging pattern captures **k-value, architecture type, training tokens, hook point, expansion factor** as params, and **reconstruction MSE, L0 sparsity, dead features percentage, explained variance** as metrics. The `infer_signature()` function generates input/output schemas automatically from sample tensors.

Modern MLflow (2.x+) replaces the legacy Staging/Production/Archived stages with **aliases**. Set `client.set_registered_model_alias("gpt2-sae-models", "champion", version=5)` to designate the production model, then load it via `mlflow.pytorch.load_model("models:/gpt2-sae-models@champion")`. This is more flexible than stages—you can define arbitrary aliases like `challenger`, `baseline`, or `experiment-v3`.

For **comparing metrics across SAE runs**, use parent/child run patterns. A parent run represents an entire sweep (e.g., "k-value grid search March 2026"), with nested child runs for each k-value configuration. The parent aggregates best metrics for cross-sweep comparison. Programmatic comparison via `mlflow.search_runs()` supports SQL-like filtering: `filter_string="params.architecture_type = 'TopK_SAE' and metrics.dead_features_pct < 10"` with ordering by any metric.

**KFP integration** follows a well-documented pattern. Each KFP component sets `MLFLOW_TRACKING_URI` and `MLFLOW_S3_ENDPOINT_URL` as environment variables (injected via K8s secrets), calls `mlflow.start_run()` within its execution, and outputs the `run_id` as a KFP artifact for downstream components. The registration component consumes this run ID to call `mlflow.register_model()` and set aliases. The recommended deployment architecture runs MLflow as a separate Kubernetes service with PostgreSQL backend store and MinIO/S3 artifact store, exposed through the Kubeflow Central Dashboard via an Istio VirtualService. **Use both KFP artifacts and MLflow artifacts**—KFP for intermediate pipeline data (preprocessed datasets, temporary files), MLflow for model artifacts and experiment tracking.

---

## GPU scheduling, monorepos, and activation caching round out the platform

**Kueue vs. Volcano** represents the key scheduling decision. Kueue is lighter (admission control only, works with existing kube-scheduler), natively integrated with Training Operator v2, and ideal for quota governance and fair-sharing. Volcano is heavier (replaces the scheduler entirely) but provides gang scheduling, NUMA-aware placement, and binpacking—better for HPC-style workloads. They can run complementarily: Kueue controls admission, Volcano handles placement. For most ML platform use cases, **Kueue alone is sufficient** and is the officially recommended path.

For **monorepo consolidation**, the book doesn't cover this topic, but the established pattern uses a structure separating `libs/` (shared utilities), `models/` (architecture code), `pipelines/` (KFP definitions), `serving/` (BentoML configs), and `infra/` (K8s manifests, Dockerfiles). **Pants is the strongest build tool for Python-heavy ML monorepos**—it provides file-level dependency management, native Docker build support, and Python-first design with lower boilerplate than Bazel. Affected-target detection (`pants --changed-since=origin/main`) enables selective CI/CD, building only images whose dependencies changed.

For **activation caching** (critical for SAE training), the proven S3-based approach stores LLM intermediate activations as raw tensors in S3, then streams them during SAE training at >1000 MB/s throughput. Key optimizations include using S3 REST API directly (not boto3), HTTP instead of HTTPS for ~40% speedup, `aiohttp` for concurrent downloads, and PyTorch `.share_memory_()` for zero-copy transfer between download processes and training. The `sache` library (github.com/Lewington-pitsos/sache) demonstrates caching **678M+ tokens of GPT-2 activations** and training a 24,576-dimensional SAE in under 30 minutes on 16GB VRAM. Feast serves a different purpose—runtime feature serving for online/offline consistency—and should be deployed alongside (not instead of) activation caching.

---

## Conclusion

The recommended architecture layers these tools in a clear separation of concerns: **KFP v2** owns the DAG (data prep → training → evaluation → registration → deployment), **Training Operator v2 + Kueue** owns GPU lifecycle (admission, scheduling, distributed training), **MLflow** owns experiment state (params, metrics, model versions, aliases), **BentoML** owns serving (LLM+SAE co-hosting, adaptive batching, multi-variant routing), and **S3/MinIO** serves as the universal artifact substrate. The Manning book's Chapters 4, 5, 6, 9, and 10 provide end-to-end implementation walkthroughs for this exact stack. The one gap is monorepo organization, which falls outside the book's scope but is best addressed with Pants. For the specific LLM+SAE use case, Pattern A (single BentoML service with multiple SAE checkpoints in a dictionary, routed by request parameter) is the pragmatic starting point, graduating to Pattern B (`bentoml.depends()` distributed services) only when independent SAE scaling becomes necessary.
