"""
Visualize Results: Generate comparison plots for SAE evaluations
"""
import os
import sys
import yaml
import fire
from pathlib import Path

import sae_bench.sae_bench_utils.graphing_utils as graphing_utils


def load_params():
    """Load parameters from params.yaml"""
    params_path = Path(__file__).parent.parent / "params.yaml"
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)


def find_eval_results(core_dir: str, sparse_probing_dir: str, absorption_dir: str,
                      scr_dir: str, tpp_dir: str, params):
    """Find all evaluation result files

    Args:
        core_dir: Directory with core evaluation results
        sparse_probing_dir: Directory with sparse probing results
        absorption_dir: Directory with absorption results
        scr_dir: Directory with SCR results
        tpp_dir: Directory with TPP results
        params: Loaded parameters (for checking enabled eval types)
    """
    print("\n=== Finding Evaluation Results ===")

    results = {}

    # Map eval types to their directories
    eval_dirs = {
        'sparse_probing': sparse_probing_dir,
        'absorption': absorption_dir,
        'scr': scr_dir,
        'tpp': tpp_dir,
    }

    # Check each eval type if enabled
    for eval_type, eval_dir in eval_dirs.items():
        # Skip if not enabled in params
        if eval_type == 'sparse_probing':
            if not params['eval_types']['sparse_probing']:
                continue
        else:
            if not params['eval_types'].get(eval_type, False):
                continue

        eval_path = Path(eval_dir)
        if eval_path.exists():
            eval_filenames = graphing_utils.find_eval_results_files([str(eval_path)])
            if eval_filenames:
                results[eval_type] = eval_filenames
                print(f"  {eval_type}: found {len(eval_filenames)} result file(s)")
        else:
            print(f"  {eval_type}: folder not found, skipping")

    # Core results (always needed for plots)
    core_path = Path(core_dir)
    if core_path.exists():
        core_filenames = graphing_utils.find_eval_results_files([str(core_path)])
        if core_filenames:
            results['core'] = core_filenames
            print(f"  core: found {len(core_filenames)} result file(s)")
    else:
        print(f"  WARNING: core folder not found at {core_dir}")

    return results


def generate_plots(eval_results, output_dir: str, params):
    """Generate comparison plots for each evaluation type

    Args:
        eval_results: Dict mapping eval types to result files
        output_dir: Directory to save generated images
        params: Loaded parameters (for visualization settings)
    """
    print("\n=== Generating Plots ===")

    image_folder = Path(output_dir)
    image_folder.mkdir(parents=True, exist_ok=True)

    # Get visualization settings
    trainer_markers = params['visualization']['trainer_markers']
    trainer_colors = params['visualization']['trainer_colors']
    k = params['visualization']['k']

    # Core results are needed for all plots
    if 'core' not in eval_results:
        print("ERROR: Core results not found. Cannot generate plots.")
        return

    core_filenames = eval_results['core']

    # Generate plots for each evaluation type
    for eval_type, eval_filenames in eval_results.items():
        if eval_type == 'core':
            continue  # Core metrics are plotted with other eval types

        print(f"\nGenerating plots for {eval_type}")

        image_base_name = str(image_folder / eval_type)

        try:
            graphing_utils.plot_results(
                eval_filenames,
                core_filenames,
                eval_type,
                image_base_name,
                k=k,
                trainer_markers=trainer_markers,
                trainer_colors=trainer_colors,
            )
            print(f"  Saved plots to {image_base_name}*.png")
        except Exception as e:
            print(f"  ERROR generating plots for {eval_type}: {e}")

    print(f"\n=== Plots Generated ===")
    print(f"Output directory: {image_folder}")


def main(
    core_dir: str,
    sparse_probing_dir: str,
    absorption_dir: str,
    scr_dir: str,
    tpp_dir: str,
    output_dir: str
):
    """Generate visualization plots for SAE evaluations

    Args:
        core_dir: Directory with core evaluation results
        sparse_probing_dir: Directory with sparse probing results
        absorption_dir: Directory with absorption results
        scr_dir: Directory with SCR results
        tpp_dir: Directory with TPP results
        output_dir: Directory to save generated images
    """
    # Load parameters
    params = load_params()

    # Find evaluation results
    eval_results = find_eval_results(
        core_dir, sparse_probing_dir, absorption_dir, scr_dir, tpp_dir, params
    )

    if not eval_results:
        print("\nNo evaluation results found. Run evaluations first.")
        sys.exit(1)

    if 'core' not in eval_results:
        print("\nERROR: Core results not found. Core evaluation is required for plotting.")
        sys.exit(1)

    # Generate plots
    generate_plots(eval_results, output_dir, params)

    print("\n=== Done ===")


if __name__ == "__main__":
    fire.Fire(main)
