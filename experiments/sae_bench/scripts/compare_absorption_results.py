#!/usr/bin/env python3
"""
Compare absorption evaluation results across different SAE variants.
Reads JSON files from eval_results/absorption/ and creates a comparison table.
"""

import json
import os
from pathlib import Path
import pandas as pd
from tabulate import tabulate


def extract_sae_id(filename):
    """Extract simplified SAE ID from filename."""
    # Remove directory path and extension
    base_name = os.path.basename(filename).replace("_eval_results.json", "")

    # Handle special cases for gemma-scope SAEs
    if "gemma-scope" in base_name:
        # Extract key info: width and type
        if "width_16k" in base_name:
            return "gemma-scope_16k"
        elif "width_65k" in base_name:
            return "gemma-scope_65k"
        else:
            return "gemma-scope"

    # For nested topk SAEs, remove "_custom_sae" suffix
    base_name = base_name.replace("_custom_sae", "")

    return base_name


def load_absorption_results(results_dir):
    """Load all absorption result JSON files."""
    results_dir = Path(results_dir)
    results = {}

    for json_file in results_dir.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)

        sae_id = extract_sae_id(str(json_file))

        # Extract metrics from eval_result_metrics.mean
        if 'eval_result_metrics' in data and 'mean' in data['eval_result_metrics']:
            metrics = data['eval_result_metrics']['mean']
            results[sae_id] = metrics
        else:
            print(f"Warning: No eval_result_metrics.mean found in {json_file}")

    return results


def create_comparison_table(results):
    """Create comparison table from results dictionary."""
    if not results:
        print("No results found!")
        return None

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')

    # Sort by mean_absorption_fraction_score (higher is better)
    if 'mean_absorption_fraction_score' in df.columns:
        df = df.sort_values('mean_absorption_fraction_score', ascending=False)

    # Round values for better readability
    df = df.round(4)

    # Add rank column
    df.insert(0, 'rank', range(1, len(df) + 1))

    return df


def display_results(df, save_csv=True):
    """Display results as formatted table and optionally save as CSV."""
    if df is None:
        return

    print("\n" + "="*80)
    print("ABSORPTION EVALUATION RESULTS COMPARISON")
    print("="*80)

    # Prepare display columns with shorter names
    display_columns = {
        'rank': 'Rank',
        'mean_absorption_fraction_score': 'Mean Abs Frac',
        'mean_full_absorption_score': 'Mean Full Abs',
        'mean_num_split_features': 'Mean Split Feat',
        'std_dev_absorption_fraction_score': 'Std Abs Frac',
        'std_dev_full_absorption_score': 'Std Full Abs',
        'std_dev_num_split_features': 'Std Split Feat'
    }

    # Rename columns for display
    display_df = df.rename(columns=display_columns)

    # Format the table
    table = tabulate(
        display_df,
        headers='keys',
        tablefmt='grid',
        floatfmt='.4f',
        showindex=True
    )

    print(table)

    # Print summary statistics
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS")
    print("-"*80)

    # Find best performers
    print(f"\nBest Mean Absorption Fraction Score:")
    best_idx = df['mean_absorption_fraction_score'].idxmax()
    print(f"  {best_idx}: {df.loc[best_idx, 'mean_absorption_fraction_score']:.4f}")

    print(f"\nBest Mean Full Absorption Score:")
    best_idx = df['mean_full_absorption_score'].idxmax()
    print(f"  {best_idx}: {df.loc[best_idx, 'mean_full_absorption_score']:.4f}")

    print(f"\nLowest Mean Split Features (better):")
    best_idx = df['mean_num_split_features'].idxmin()
    print(f"  {best_idx}: {df.loc[best_idx, 'mean_num_split_features']:.4f}")

    # Save to CSV
    if save_csv:
        csv_path = Path(__file__).parent.parent / "eval_results" / "absorption_comparison.csv"
        df.to_csv(csv_path)
        print(f"\nResults saved to: {csv_path}")

    print("\n" + "="*80)

    # Also create a simpler comparison focusing on key metrics
    print("\n" + "="*80)
    print("SIMPLIFIED COMPARISON (Key Metrics Only)")
    print("="*80)

    simple_df = df[['rank', 'mean_absorption_fraction_score', 'mean_full_absorption_score', 'mean_num_split_features']]
    simple_df = simple_df.rename(columns={
        'rank': 'Rank',
        'mean_absorption_fraction_score': 'Absorption Fraction',
        'mean_full_absorption_score': 'Full Absorption',
        'mean_num_split_features': 'Split Features'
    })

    simple_table = tabulate(
        simple_df,
        headers='keys',
        tablefmt='grid',
        floatfmt='.4f',
        showindex=True
    )

    print(simple_table)

    return df


def main():
    """Main function to run absorption comparison."""
    # Get the eval_results/absorption directory
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "eval_results" / "absorption"

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print(f"Loading absorption results from: {results_dir}")

    # Load results
    results = load_absorption_results(results_dir)

    if not results:
        print("No absorption results found!")
        return

    print(f"Found {len(results)} absorption result files")

    # Create comparison table
    df = create_comparison_table(results)

    # Display results
    display_results(df, save_csv=True)

    return df


if __name__ == "__main__":
    main()