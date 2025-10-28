"""
Process Results: Merge core metrics with eval-specific metrics
"""
import os
import sys
import yaml
import json
import fire
from pathlib import Path

import sae_bench.sae_bench_utils.graphing_utils as graphing_utils


def load_params():
    """Load parameters from params.yaml"""
    params_path = Path(__file__).parent.parent / "params.yaml"
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)


def find_eval_results(eval_dir: str):
    """Find all evaluation result files in a directory"""
    eval_path = Path(eval_dir)

    if not eval_path.exists():
        print(f"  WARNING: {eval_dir} not found, skipping")
        return []

    eval_filenames = graphing_utils.find_eval_results_files([str(eval_path)])
    if eval_filenames:
        print(f"  Found {len(eval_filenames)} result file(s) in {eval_dir}")
    return eval_filenames


def merge_results(eval_results_files, core_results_files):
    """Merge eval-specific metrics with core metrics"""
    print("\n=== Merging Results ===")

    # Load evaluation results
    eval_results_dict = graphing_utils.get_eval_results(eval_results_files)
    print(f"Loaded evaluation results for {len(eval_results_dict)} SAE(s)")

    # Load core results
    core_results_dict = graphing_utils.get_eval_results(core_results_files)
    print(f"Loaded core results for {len(core_results_dict)} SAE(s)")

    # Merge core metrics into evaluation results
    for sae in eval_results_dict:
        if sae in core_results_dict:
            eval_results_dict[sae].update(core_results_dict[sae])
            print(f"  Merged results for: {sae}")
        else:
            print(f"  WARNING: No core results for {sae}")

    return eval_results_dict


def save_merged_results(merged_results, output_dir: str):
    """Save merged results to output folder"""
    print(f"\n=== Saving Merged Results to {output_dir} ===")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "merged_results.json"

    with open(output_file, 'w') as f:
        json.dump(merged_results, f, indent=2)

    print(f"Saved merged results to: {output_file}")

    # Save individual SAE results for easier access
    for sae_name, sae_results in merged_results.items():
        sae_file = output_path / f"{sae_name}_results.json"
        with open(sae_file, 'w') as f:
            json.dump({sae_name: sae_results}, f, indent=2)
        print(f"  Saved {sae_name} results")


def main(
    core_dir: str,
    sparse_probing_dir: str,
    absorption_dir: str,
    scr_dir: str,
    tpp_dir: str,
    output_dir: str
):
    """Process and merge evaluation results

    Args:
        core_dir: Directory with core evaluation results
        sparse_probing_dir: Directory with sparse probing results
        absorption_dir: Directory with absorption results
        scr_dir: Directory with SCR results
        tpp_dir: Directory with TPP results
        output_dir: Directory to save merged results
    """
    # Load parameters
    params = load_params()

    print("\n=== Finding Evaluation Results ===")

    # Find core results (required)
    core_results_files = find_eval_results(core_dir)
    if not core_results_files:
        print("\nERROR: Core results not found. Core evaluation is required.")
        sys.exit(1)

    # Find eval-specific results based on what's enabled
    eval_results = {}

    if params['eval_types']['sparse_probing']:
        files = find_eval_results(sparse_probing_dir)
        if files:
            eval_results['sparse_probing'] = files

    if params['eval_types'].get('absorption', False):
        files = find_eval_results(absorption_dir)
        if files:
            eval_results['absorption'] = files

    if params['eval_types'].get('scr', False):
        files = find_eval_results(scr_dir)
        if files:
            eval_results['scr'] = files

    if params['eval_types'].get('tpp', False):
        files = find_eval_results(tpp_dir)
        if files:
            eval_results['tpp'] = files

    # Merge results for each evaluation type
    all_merged_results = {}

    for eval_type, eval_files in eval_results.items():
        print(f"\n--- Processing {eval_type} results ---")
        merged = merge_results(eval_files, core_results_files)

        # Update all_merged_results with this eval type's metrics
        for sae_name, sae_results in merged.items():
            if sae_name not in all_merged_results:
                all_merged_results[sae_name] = {}
            all_merged_results[sae_name][eval_type] = sae_results

    # Also save core-only results
    print(f"\n--- Processing core-only results ---")
    core_only = graphing_utils.get_eval_results(core_results_files)
    for sae_name, sae_results in core_only.items():
        if sae_name not in all_merged_results:
            all_merged_results[sae_name] = {}
        all_merged_results[sae_name]['core'] = sae_results

    # Save merged results
    save_merged_results(all_merged_results, output_dir)

    print("\n=== Done ===")


if __name__ == "__main__":
    fire.Fire(main)
