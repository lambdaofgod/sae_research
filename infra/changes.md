# MLflow Tracking — Changes

## Dependencies
- Added `mlflow`
- Removed `trackio` (WandB)

## New files
- **`sae_research/training/mlflow_logging.py`** — Thin wrapper for parent/child MLflow runs, metric logging, artifact logging
- **`infra/activault/activault/templates/statefulset-postgres.yaml`** — PostgreSQL StatefulSet (10Gi PVC) + Service
- **`infra/activault/activault/templates/deployment-mlflow.yaml`** — MLflow server Deployment + Service (postgres backend, Garage S3 artifacts)
- **`infra/activault/activault/templates/secret-mlflow.yaml`** — Auto-generated DB credentials
- **`test/test_mlflow_training_smoke.py`** — Smoke test: local file-backed MLflow server, 5-step CPU training, asserts parent/child runs, metrics, and artifacts

## Modified files
- **`sae_research/training/train.py`** — Removed all WandB code (`trackio` import, `new_wandb_process()`, `mp.Queue`/`mp.Process` plumbing). MLflow logs directly in the training loop. `trainSAE()` takes `use_mlflow`/`mlflow_parent_run_id`, returns child run IDs.
- **`sae_research/training/runner.py`** — `--use_wandb` → `--use_mlflow`. Creates parent MLflow sweep run, passes run ID to `trainSAE()`, logs eval metrics back to child runs. Removed `mp.set_start_method("spawn")`.
- **`sae_research/training/config.py`** — Removed `wandb_name` from `BaseTrainerConfig` and all `get_trainer_configs()` calls. `wandb_project` → `mlflow_experiment`.
- **`sae_research/training/configs/defaults.yaml`** — `wandb_project` → `mlflow_experiment: sae_training`
- **`infra/activault/activault/values.yaml`** — Added `postgres:` and `mlflow:` sections, proxy entry on port 8085
- **`infra/activault/activault/templates/job-garage-bootstrap.yaml`** — Refactored bucket creation into `ensure_bucket()`, creates `mlflow-artifacts` bucket alongside `activations`
- **`infra/TASKS.org`** — Marked MLflow subtasks DONE, promoted KFP evaluation pipeline to top-level task with 4 subtasks
- **`pyproject.toml`** — mlflow added, trackio removed
