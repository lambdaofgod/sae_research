#!/usr/bin/env python3
"""Extract SAEBench evaluation metrics from results for DVC tracking."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List


def extract_core_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from core evaluation results."""
    metrics = {}
    result_metrics = data.get('eval_result_metrics', {})

    # Model preservation metrics
    if 'model_behavior_preservation' in result_metrics:
        mbp = result_metrics['model_behavior_preservation']
        metrics['kl_div_score'] = mbp.get('kl_div_score')

    if 'model_performance_preservation' in result_metrics:
        mpp = result_metrics['model_performance_preservation']
        metrics['ce_loss_score'] = mpp.get('ce_loss_score')

    # Reconstruction quality metrics
    if 'reconstruction_quality' in result_metrics:
        rq = result_metrics['reconstruction_quality']
        metrics['explained_variance'] = rq.get('explained_variance')
        metrics['mse'] = rq.get('mse')
        metrics['cossim'] = rq.get('cossim')

    # Sparsity metrics
    if 'sparsity' in result_metrics:
        sp = result_metrics['sparsity']
        metrics['l0'] = sp.get('l0')
        metrics['l1'] = sp.get('l1')

    # Shrinkage metrics
    if 'shrinkage' in result_metrics:
        sh = result_metrics['shrinkage']
        metrics['l2_ratio'] = sh.get('l2_ratio')
        metrics['relative_reconstruction_bias'] = sh.get('relative_reconstruction_bias')

    # Misc metrics
    if 'misc_metrics' in result_metrics:
        misc = result_metrics['misc_metrics']
        metrics['frac_alive'] = misc.get('frac_alive')
        metrics['freq_over_1_percent'] = misc.get('freq_over_1_percent')

    return metrics


def extract_sparse_probing_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from sparse probing evaluation results."""
    metrics = {}
    result_metrics = data.get('eval_result_metrics', {})

    # SAE metrics
    if 'sae' in result_metrics:
        sae = result_metrics['sae']
        metrics['sae_test_accuracy'] = sae.get('sae_test_accuracy')
        for k in [1, 2, 5, 10, 20, 50, 100]:
            key = f'sae_top_{k}_test_accuracy'
            if key in sae:
                metrics[key] = sae[key]

    # LLM metrics
    if 'llm' in result_metrics:
        llm = result_metrics['llm']
        metrics['llm_test_accuracy'] = llm.get('llm_test_accuracy')
        for k in [1, 2, 5, 10, 20, 50, 100]:
            key = f'llm_top_{k}_test_accuracy'
            if key in llm:
                metrics[key] = llm[key]

    return metrics


def extract_scr_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from SCR (Spurious Correlation Removal) results."""
    metrics = {}
    result_metrics = data.get('eval_result_metrics', {})

    if 'scr_metrics' in result_metrics:
        scr = result_metrics['scr_metrics']
        # Extract metrics for different thresholds
        for threshold in [2, 5, 10, 20, 50, 100, 500]:
            metric_key = f'scr_metric_threshold_{threshold}'
            if metric_key in scr:
                metrics[metric_key] = scr[metric_key]

            # Also extract directional metrics
            dir1_key = f'scr_dir1_threshold_{threshold}'
            dir2_key = f'scr_dir2_threshold_{threshold}'
            if dir1_key in scr:
                metrics[dir1_key] = scr[dir1_key]
            if dir2_key in scr:
                metrics[dir2_key] = scr[dir2_key]

    return metrics


def extract_tpp_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from TPP (Targeted Probe Perturbation) results."""
    metrics = {}
    result_metrics = data.get('eval_result_metrics', {})

    if 'tpp_metrics' in result_metrics:
        tpp = result_metrics['tpp_metrics']
        # Extract metrics for different thresholds
        for threshold in [2, 5, 10, 20, 50, 100, 500]:
            total_key = f'tpp_threshold_{threshold}_total_metric'
            intended_key = f'tpp_threshold_{threshold}_intended_diff_only'
            unintended_key = f'tpp_threshold_{threshold}_unintended_diff_only'

            if total_key in tpp:
                metrics[total_key] = tpp[total_key]
            if intended_key in tpp:
                metrics[intended_key] = tpp[intended_key]
            if unintended_key in tpp:
                metrics[unintended_key] = tpp[unintended_key]

    return metrics


def extract_absorption_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from absorption evaluation results."""
    metrics = {}
    result_metrics = data.get('eval_result_metrics', {})

    # Absorption metrics structure may vary, extract what's available
    if 'absorption_metrics' in result_metrics:
        abs_metrics = result_metrics['absorption_metrics']
        # Add specific absorption metrics when structure is known
        for key, value in abs_metrics.items():
            if isinstance(value, (int, float)):
                metrics[key] = value

    return metrics


def extract_unlearning_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from unlearning evaluation results."""
    metrics = {}
    result_metrics = data.get('eval_result_metrics', {})

    # Unlearning metrics structure may vary, extract what's available
    if 'unlearning_metrics' in result_metrics:
        unlearn = result_metrics['unlearning_metrics']
        # Add specific unlearning metrics when structure is known
        for key, value in unlearn.items():
            if isinstance(value, (int, float)):
                metrics[key] = value

    return metrics


def process_results_directory(
    results_dir: str,
    metric_type: str,
    output_file: str
) -> None:
    """Process all JSON files in a results directory and extract metrics.

    Args:
        results_dir: Directory containing evaluation JSON results
        metric_type: Type of metrics to extract (core, sparse_probing, scr, tpp, absorption, unlearning)
        output_file: Output file path for extracted metrics
    """
    results_path = Path(results_dir)
    all_metrics = {}

    if not results_path.exists():
        print(f"Results directory {results_dir} does not exist. Creating empty metrics file.")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        return

    # Select extraction function based on metric type
    extractors = {
        'core': extract_core_metrics,
        'sparse_probing': extract_sparse_probing_metrics,
        'scr': extract_scr_metrics,
        'tpp': extract_tpp_metrics,
        'absorption': extract_absorption_metrics,
        'unlearning': extract_unlearning_metrics,
    }

    if metric_type not in extractors:
        print(f"Unknown metric type: {metric_type}")
        sys.exit(1)

    extractor = extractors[metric_type]

    # Process each JSON file in the results directory
    json_files = sorted(results_path.glob("*.json"))

    for json_file in json_files:
        # Extract SAE name from filename (remove _eval_results.json suffix)
        sae_name = json_file.stem.replace("_eval_results", "")

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract metrics using appropriate extractor
            metrics = extractor(data)

            if metrics:
                all_metrics[sae_name] = metrics
                print(f"Extracted {metric_type} metrics from {json_file.name}")

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Save metrics to output file
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Saved {metric_type} metrics to {output_file}")

    # Print summary
    if all_metrics:
        print(f"\nProcessed {len(all_metrics)} SAE(s)")
        # Print first few metrics as example
        for sae_name in list(all_metrics.keys())[:2]:
            print(f"  {sae_name}: {len(all_metrics[sae_name])} metrics")


def aggregate_all_metrics(metric_files: List[str], output_file: str) -> None:
    """Aggregate metrics from multiple metric files into a single file.

    Args:
        metric_files: List of metric JSON files to aggregate
        output_file: Output file path for aggregated metrics
    """
    aggregated = {}

    for metric_file in metric_files:
        metric_path = Path(metric_file)
        if not metric_path.exists():
            print(f"Warning: {metric_file} does not exist, skipping...")
            continue

        # Extract metric type from filename (e.g., "core" from "metrics/core.json")
        metric_type = metric_path.stem

        try:
            with open(metric_path, 'r') as f:
                metrics = json.load(f)

            # Merge metrics, organizing by SAE name
            for sae_name, sae_metrics in metrics.items():
                if sae_name not in aggregated:
                    aggregated[sae_name] = {}

                # Add metrics under the metric type namespace
                aggregated[sae_name][metric_type] = sae_metrics

        except Exception as e:
            print(f"Error reading {metric_file}: {e}")
            continue

    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Save aggregated metrics
    with open(output_file, 'w') as f:
        json.dump(aggregated, f, indent=2)

    print(f"Aggregated metrics saved to {output_file}")
    print(f"Total SAEs: {len(aggregated)}")


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Extract specific metrics: python extract_metrics.py <metric_type> <results_dir> <output_file>")
        print("    metric_type: core, sparse_probing, scr, tpp, absorption, unlearning")
        print("  Aggregate metrics: python extract_metrics.py aggregate <metric_file1> <metric_file2> ... <output_file>")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'aggregate':
        if len(sys.argv) < 4:
            print("Usage: python extract_metrics.py aggregate <metric_file1> <metric_file2> ... <output_file>")
            sys.exit(1)

        metric_files = sys.argv[2:-1]
        output_file = sys.argv[-1]
        aggregate_all_metrics(metric_files, output_file)

    elif command in ['core', 'sparse_probing', 'scr', 'tpp', 'absorption', 'unlearning']:
        if len(sys.argv) != 4:
            print(f"Usage: python extract_metrics.py {command} <results_dir> <output_file>")
            sys.exit(1)

        results_dir = sys.argv[2]
        output_file = sys.argv[3]
        process_results_directory(results_dir, command, output_file)

    else:
        print(f"Unknown command: {command}")
        print("Valid commands: core, sparse_probing, scr, tpp, absorption, unlearning, aggregate")
        sys.exit(1)


if __name__ == "__main__":
    main()