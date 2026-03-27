"""
Standalone eval components that can run outside KFP.

Each function wraps a sae_bench evaluation and can be called directly
or later wrapped as a @dsl.component for KFP.

All functions that log to MLflow require explicit tags to prevent
cross-contamination between experiments. At minimum, pass an
experiment name tag so runs are always attributable.
"""

import mlflow
import sae_bench.evals.core.main as core_eval_module


def log_eval_to_mlflow(
    run_id: str,
    metrics: dict,
    tags: dict[str, str],
):
    """Log eval metrics and tags to an existing MLflow run.

    Args:
        run_id: MLflow run ID to log to (typically the training child run).
        metrics: Flat dict of metric_name -> float.
        tags: Required tags. Must include at least 'experiment_name'.
    """
    if "experiment_name" not in tags:
        raise ValueError(
            "tags must include 'experiment_name' to prevent orphaned eval runs. "
            "Example: {'experiment_name': 'topk_vs_batchtopk_march2024'}"
        )

    with mlflow.start_run(run_id=run_id, nested=True):
        mlflow.set_tags(tags)
        mlflow.log_metrics(metrics)


def run_core_eval(
    selected_saes: list[tuple],
    n_reconstruction_batches: int = 2000,
    n_sparsity_batches: int = 2000,
    batch_size: int = 4,
    dataset: str = "Skylion007/openwebtext",
    context_size: int = 128,
    dtype: str = "float32",
    device: str = "cuda",
    output_folder: str = "eval_results/core",
) -> list[dict]:
    """Run sae_bench core evaluation (KL div, CE loss, explained variance, L0, etc.).

    Args:
        selected_saes: List of (name, sae) tuples. SAEs must be BaseSAE subclasses
            or sae_lens SAE objects.
        n_reconstruction_batches: Number of batches for reconstruction metrics.
        n_sparsity_batches: Number of batches for sparsity/variance metrics.
        batch_size: Number of prompts per batch.
        dataset: HuggingFace dataset for evaluation.
        context_size: Token context length.
        dtype: Data type string (float32, bfloat16).
        device: Device string.
        output_folder: Where sae_bench writes result JSONs.

    Returns:
        List of result dicts, one per SAE.
    """
    results = core_eval_module.multiple_evals(
        selected_saes=selected_saes,
        n_eval_reconstruction_batches=n_reconstruction_batches,
        n_eval_sparsity_variance_batches=n_sparsity_batches,
        eval_batch_size_prompts=batch_size,
        compute_featurewise_density_statistics=True,
        compute_featurewise_weight_based_metrics=True,
        exclude_special_tokens_from_reconstruction=True,
        dataset=dataset,
        context_size=context_size,
        output_folder=output_folder,
        verbose=True,
        dtype=dtype,
        device=device,
    )
    return results


def extract_core_metrics(result: dict) -> dict:
    """Extract flat metrics from a sae_bench core eval result dict.

    Args:
        result: Single SAE result from run_core_eval.

    Returns:
        Flat dict of metric_name -> float, prefixed with 'core_'.
    """
    metrics = {}
    # multiple_evals returns "metrics", saved JSON files use "eval_result_metrics"
    rm = result.get("metrics") or result.get("eval_result_metrics", {})

    if "model_behavior_preservation" in rm:
        mbp = rm["model_behavior_preservation"]
        if "kl_div_score" in mbp:
            metrics["core_kl_div_score"] = mbp["kl_div_score"]

    if "model_performance_preservation" in rm:
        mpp = rm["model_performance_preservation"]
        for k in ["ce_loss_score", "ce_loss_with_sae"]:
            if k in mpp:
                metrics[f"core_{k}"] = mpp[k]

    if "reconstruction_quality" in rm:
        rq = rm["reconstruction_quality"]
        for k in ["explained_variance", "mse", "cossim"]:
            if k in rq:
                metrics[f"core_{k}"] = rq[k]

    if "sparsity" in rm:
        sp = rm["sparsity"]
        for k in ["l0", "l1"]:
            if k in sp:
                metrics[f"core_{k}"] = sp[k]

    if "shrinkage" in rm:
        sh = rm["shrinkage"]
        for k in ["l2_ratio", "relative_reconstruction_bias"]:
            if k in sh:
                metrics[f"core_{k}"] = sh[k]

    if "misc_metrics" in rm:
        misc = rm["misc_metrics"]
        for k in ["frac_alive"]:
            if k in misc:
                metrics[f"core_{k}"] = misc[k]

    return metrics
