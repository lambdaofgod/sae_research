"""
Eval reconciler: discovers missing (SAE, eval_type) pairs and runs evaluations.

Follows the reconciler pattern:
- Desired state: eval types that every SAE should have (from pipeline params)
- Current state: MLflow metric keys on each SAE's run
- Gap: (SAE, eval_type) pairs where metrics are missing

Can be used standalone, from notebooks, or inside a KFP component.
"""

import shutil
import tempfile

import mlflow
import torch

from sae_research.eval.sae_wrapper import load_sae_for_eval
from sae_research.eval.components import (
    run_core_eval,
    extract_core_metrics,
    log_eval_to_mlflow,
)


# Map eval type names to the metric prefix used to detect presence in MLflow
EVAL_TYPE_METRIC_PREFIX = {
    "core": "core_",
    # Future eval types:
    # "sparse_probing": "sparse_probing_",
    # "absorption": "absorption_",
    # "scr": "scr_",
    # "tpp": "tpp_",
}


def discover_gaps(
    tracking_uri: str,
    experiment_name: str,
    eval_types: list[str],
    run_ids: list[str] | None = None,
    force: bool = False,
) -> list[dict]:
    """Query MLflow and return list of (run_id, eval_type) gaps to fill.

    Args:
        tracking_uri: MLflow tracking server URI.
        experiment_name: MLflow experiment name to search.
        eval_types: Eval types to check (e.g. ["core"]).
        run_ids: Only check these runs. None = all SAE runs in experiment.
        force: If True, return all pairs even if metrics already exist.

    Returns:
        List of dicts with keys: run_id, eval_type, model_name, layer.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    for et in eval_types:
        if et not in EVAL_TYPE_METRIC_PREFIX:
            raise ValueError(
                f"Unknown eval type: {et}. Known: {list(EVAL_TYPE_METRIC_PREFIX.keys())}"
            )

    if run_ids:
        runs = [client.get_run(rid) for rid in run_ids]
    else:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found in MLflow")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.trainer_class != ''",
        )

    gaps = []
    for run in runs:
        for eval_type in eval_types:
            prefix = EVAL_TYPE_METRIC_PREFIX[eval_type]
            has_metrics = any(k.startswith(prefix) for k in run.data.metrics)
            if force or not has_metrics:
                gaps.append(
                    {
                        "run_id": run.info.run_id,
                        "eval_type": eval_type,
                        "model_name": run.data.params.get("lm_name", ""),
                        "layer": int(run.data.params.get("layer", "0")),
                    }
                )

    return gaps


def run_eval_for_gap(
    tracking_uri: str,
    gap: dict,
    experiment_name: str,
    device: str = "cuda",
    dtype: str = "float32",
    output_folder: str = "/tmp/eval_results",
    eval_config: dict | None = None,
) -> dict:
    """Run a single eval type for a single SAE run.

    Downloads the SAE artifacts from MLflow, runs the eval, logs results back.

    Args:
        eval_config: Optional overrides for eval settings (e.g.
            n_reconstruction_batches, n_sparsity_batches, batch_size,
            context_size). Passed through to run_core_eval.

    Returns:
        Dict of metrics that were logged.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    run_id = gap["run_id"]
    eval_type = gap["eval_type"]
    model_name = gap["model_name"]
    layer = gap["layer"]
    eval_kwargs = eval_config or {}

    # Download SAE artifacts from MLflow
    artifact_dir = tempfile.mkdtemp(prefix=f"eval_{run_id[:8]}_")
    try:
        for artifact_name in ["ae.pt", "config.json"]:
            client.download_artifacts(run_id, artifact_name, artifact_dir)

        sae_name, wrapped_sae = load_sae_for_eval(
            sae_path=artifact_dir,
            model_name=model_name,
            hook_layer=layer,
            device=device,
            dtype=getattr(torch, dtype),
        )

        if eval_type == "core":
            results = run_core_eval(
                selected_saes=[(sae_name, wrapped_sae)],
                dtype=dtype,
                device=device,
                output_folder=output_folder,
                **eval_kwargs,
            )
            metrics = extract_core_metrics(results[0])
        else:
            raise ValueError(f"Eval type '{eval_type}' not yet implemented")

        log_eval_to_mlflow(
            run_id=run_id,
            metrics=metrics,
            tags={
                "experiment_name": experiment_name,
                "eval_type": eval_type,
            },
        )

        return metrics

    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


def reconcile(
    tracking_uri: str,
    experiment_name: str,
    eval_types: list[str] | None = None,
    run_ids: list[str] | None = None,
    force: bool = False,
    device: str = "cuda",
    dtype: str = "float32",
    output_folder: str = "/tmp/eval_results",
    eval_config: dict | None = None,
) -> list[dict]:
    """Full reconciliation: discover gaps, run evals sequentially, return results.

    This is the main entry point. It can also be used for targeted reruns by
    passing specific run_ids and/or eval_types, with force=True.

    Args:
        tracking_uri: MLflow tracking server URI.
        experiment_name: Required — used as tag on all eval runs.
        eval_types: Eval types to reconcile. None = all known types.
        run_ids: Only these SAE runs. None = all SAE runs in experiment.
        force: Rerun even if metrics already exist.
        device: Device for eval (cuda, cpu).
        dtype: Data type for eval (float32, bfloat16).
        output_folder: Where sae_bench writes intermediate result JSONs.
        eval_config: Optional overrides for eval settings (e.g.
            n_reconstruction_batches, n_sparsity_batches, batch_size,
            context_size). Passed through to run_core_eval.

    Returns:
        List of {run_id, eval_type, metrics} for each completed eval.
    """
    if eval_types is None:
        eval_types = list(EVAL_TYPE_METRIC_PREFIX.keys())

    gaps = discover_gaps(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        eval_types=eval_types,
        run_ids=run_ids,
        force=force,
    )

    print(f"Found {len(gaps)} eval gaps to fill")

    completed = []
    for i, gap in enumerate(gaps):
        print(
            f"\n[{i + 1}/{len(gaps)}] {gap['eval_type']} on run {gap['run_id'][:8]}..."
        )
        metrics = run_eval_for_gap(
            tracking_uri=tracking_uri,
            gap=gap,
            experiment_name=experiment_name,
            device=device,
            dtype=dtype,
            output_folder=output_folder,
            eval_config=eval_config,
        )
        completed.append(
            {
                "run_id": gap["run_id"],
                "eval_type": gap["eval_type"],
                "metrics": metrics,
            }
        )
        print(f"  Logged {len(metrics)} metrics")

    print(f"\nReconciliation complete: {len(completed)}/{len(gaps)} evals run")
    return completed
