#!/usr/bin/env python3
"""Generate METRICS.md report from evaluation results using pandas."""

import json
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import yaml


def load_metrics(metrics_file: str = "metrics/all_metrics.json") -> Dict[str, Any]:
    """Load aggregated metrics from JSON file."""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def load_params(params_file: str = "params.yaml") -> Dict[str, Any]:
    """Load parameters from YAML file."""
    with open(params_file, 'r') as f:
        return yaml.safe_load(f)


def create_core_metrics_table(metrics: Dict[str, Any]) -> str:
    """Create core metrics comparison table."""
    rows = []

    for sae_name, sae_data in metrics.items():
        if 'core' not in sae_data:
            continue

        core = sae_data['core']
        row = {
            'SAE Variant': sae_name,
            'KL Div': round(core.get('kl_div_score', 0), 3),
            'CE Loss': round(core.get('ce_loss_score', 0), 3),
            'Explained Var': round(core.get('explained_variance', 0), 3),
            'MSE': round(core.get('mse', 0), 1),
            'CosSim': round(core.get('cossim', 0), 3),
            'L0': round(core.get('l0', 0), 1),
            'L1': round(core.get('l1', 0), 0),
            'L2 Ratio': round(core.get('l2_ratio', 0), 3),
        }
        rows.append(row)

    if not rows:
        return "*No core metrics available*"

    df = pd.DataFrame(rows)
    df = df.set_index('SAE Variant')

    # Convert to markdown with proper alignment
    return df.to_markdown()


def create_sparse_probing_table(metrics: Dict[str, Any]) -> str:
    """Create sparse probing comparison table with LLM baseline."""
    rows = []
    llm_baseline = None

    for sae_name, sae_data in metrics.items():
        if 'sparse_probing' not in sae_data:
            continue

        sp = sae_data['sparse_probing']

        # Extract LLM baseline (same for all SAEs, so just take first)
        if llm_baseline is None and sp.get('llm_test_accuracy') is not None:
            llm_baseline = {
                'SAE Variant': 'LLM Baseline',
                'Test Accuracy': round(sp.get('llm_test_accuracy', 0), 3),
                'Top-1': round(sp.get('llm_top_1_test_accuracy', 0), 3),
                'Top-2': round(sp.get('llm_top_2_test_accuracy', 0), 3),
                'Top-5': round(sp.get('llm_top_5_test_accuracy', 0), 3),
            }

        # Add SAE row
        row = {
            'SAE Variant': sae_name,
            'Test Accuracy': round(sp.get('sae_test_accuracy', 0), 3),
            'Top-1': round(sp.get('sae_top_1_test_accuracy', 0), 3),
            'Top-2': round(sp.get('sae_top_2_test_accuracy', 0), 3),
            'Top-5': round(sp.get('sae_top_5_test_accuracy', 0), 3),
        }
        rows.append(row)

    if not rows:
        return "*No sparse probing metrics available*"

    # Add LLM baseline as first row
    if llm_baseline:
        rows.insert(0, llm_baseline)

    df = pd.DataFrame(rows)
    df = df.set_index('SAE Variant')

    return df.to_markdown()


def create_scr_tpp_table(metrics: Dict[str, Any]) -> str:
    """Create combined SCR & TPP metrics table."""
    rows = []

    for sae_name, sae_data in metrics.items():
        scr = sae_data.get('scr', {})
        tpp = sae_data.get('tpp', {})

        # Only include if at least one metric type is available
        if not scr and not tpp:
            continue

        row = {
            'SAE Variant': sae_name,
            'SCR (thresh=10)': round(scr.get('scr_metric_threshold_10', 0), 3) if scr else None,
            'TPP Total (thresh=10)': round(tpp.get('tpp_threshold_10_total_metric', 0), 3) if tpp else None,
            'TPP Intended': round(tpp.get('tpp_threshold_10_intended_diff_only', 0), 3) if tpp else None,
            'TPP Unintended': round(tpp.get('tpp_threshold_10_unintended_diff_only', 0), 3) if tpp else None,
        }
        rows.append(row)

    if not rows:
        return "*No SCR/TPP metrics available*"

    df = pd.DataFrame(rows)
    df = df.set_index('SAE Variant')

    return df.to_markdown()


def create_eval_config(params: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    """Create evaluation configuration section."""
    model_name = params.get('model', {}).get('name', 'unknown')
    baseline_sae = params.get('baseline_sae', {})
    hook_layer = baseline_sae.get('hook_layer', 'unknown')
    training_tokens = baseline_sae.get('training_tokens', 'unknown')

    # Get dict size from first SAE config if available
    dict_size = 'unknown'
    for sae_data in metrics.values():
        if 'core' in sae_data:
            # Try to infer from data - typically d_sae in the config
            # For now just use placeholder
            dict_size = '4608'  # Can be extracted from sae_cfg_dict if available
            break

    num_variants = len(metrics)

    # Format training tokens
    if isinstance(training_tokens, int):
        tokens_str = f"{training_tokens:,}"
    else:
        tokens_str = str(training_tokens)

    config = f"""- **Model**: {model_name}
- **Layer**: {hook_layer} (resid_post)
- **Dictionary size**: {dict_size} (2x expansion)
- **Context length**: 128
- **Training tokens**: {tokens_str}
- **SAE Variants Tested**: {num_variants}"""

    return config


def generate_report(
    template_file: str = "METRICS_TEMPLATE.md",
    output_file: str = "METRICS.md",
    metrics_file: str = "metrics/all_metrics.json",
    params_file: str = "params.yaml"
) -> None:
    """Generate METRICS.md report from template and data."""

    # Load data
    print(f"Loading metrics from {metrics_file}...")
    metrics = load_metrics(metrics_file)

    print(f"Loading params from {params_file}...")
    params = load_params(params_file)

    # Create tables
    print("Creating core metrics table...")
    core_table = create_core_metrics_table(metrics)

    print("Creating sparse probing table...")
    sparse_probing_table = create_sparse_probing_table(metrics)

    print("Creating SCR & TPP table...")
    scr_tpp_table = create_scr_tpp_table(metrics)

    print("Creating eval config...")
    eval_config = create_eval_config(params, metrics)

    # Load template
    print(f"Loading template from {template_file}...")
    with open(template_file, 'r') as f:
        template = f.read()

    # Fill template
    print("Filling template...")
    report = template.replace('{EVAL_CONFIG}', eval_config)
    report = report.replace('{CORE_METRICS_TABLE}', core_table)
    report = report.replace('{SPARSE_PROBING_TABLE}', sparse_probing_table)
    report = report.replace('{SCR_TPP_TABLE}', scr_tpp_table)

    # Write output
    print(f"Writing report to {output_file}...")
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"✓ Report generated successfully!")
    print(f"  - Processed {len(metrics)} SAE variants")
    print(f"\nNext steps:")
    print(f"  1. Edit {output_file} to rename SAE variants to friendly names")
    print(f"  2. Add **bold** to highlight best values")
    print(f"  3. Fill in 'Key Observations' sections")
    print(f"  4. Fill in plot interpretation and summary")


def main():
    """Main entry point."""
    # Use default paths relative to experiments/sae_bench/
    generate_report()


if __name__ == "__main__":
    main()
