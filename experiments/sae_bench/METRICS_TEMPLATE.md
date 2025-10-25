# SAEBench Evaluation Metrics

## Overview

This document defines the metrics tracked by the SAEBench evaluation pipeline and presents the latest evaluation results.

## Latest Evaluation Results

### Evaluation Configuration
{EVAL_CONFIG}

### Core Metrics

{CORE_METRICS_TABLE}

**Key Observations**:
- [Manual interpretation goes here after generation]

### Sparse Probing Results

{SPARSE_PROBING_TABLE}

**Key Observations**:
- [Manual interpretation goes here after generation]

**Sparsity vs Interpretability Trade-off**:

![L0 vs Sparse Probing Top-1 Accuracy by SAE Type](results/images/sparse_probing_2var_sae_type.png)

- [Manual plot interpretation goes here after generation]

### SCR & TPP Metrics

{SCR_TPP_TABLE}

**Key Observations**:
- [Manual interpretation goes here after generation]

### Absorption Metrics

{ABSORPTION_TABLE}

**Key Observations**:
- [Manual interpretation goes here after generation]

### Summary

[Manual summary goes here after generation]

## Tracked Metrics

### Core Evaluation Metrics (`results/core/*.json`)
- **Model Preservation**
  - `kl_div_score`: KL divergence score
  - `ce_loss_score`: Cross-entropy loss score
- **Reconstruction Quality**
  - `explained_variance`: Proportion of variance explained
  - `mse`: Mean squared error
  - `cossim`: Cosine similarity
- **Sparsity**
  - `l0`: Average L0 norm (number of active features)
  - `l1`: Average L1 norm
- **Shrinkage**
  - `l2_ratio`: Ratio of output to input L2 norms

### Sparse Probing Metrics (`results/sparse_probing/*.json`)
- `sae_test_accuracy`: Overall SAE test accuracy
- `llm_test_accuracy`: Overall LLM test accuracy
- `sae_top_k_test_accuracy`: Top-k accuracies for k={1,2,5,10,20,50,100}
- `llm_top_k_test_accuracy`: Top-k accuracies for k={1,2,5,10,20,50,100}

### Spurious Correlation Removal (SCR) Metrics (`results/scr/*.json`)
- `scr_metric_threshold_X`: SCR score for threshold X (X={2,5,10,20,50,100,500})
- `scr_dir1_threshold_X`: Directional metric 1 for threshold X
- `scr_dir2_threshold_X`: Directional metric 2 for threshold X

### Targeted Probe Perturbation (TPP) Metrics (`results/tpp/*.json`)
- `tpp_threshold_X_total_metric`: Total TPP metric for threshold X
- `tpp_threshold_X_intended_diff_only`: Intended differences for threshold X
- `tpp_threshold_X_unintended_diff_only`: Unintended differences for threshold X

### Absorption Metrics (`results/absorption/*.json`)
- `mean_absorption_fraction_score`: Mean absorption fraction score (higher is better)
- `mean_full_absorption_score`: Mean full absorption score
- `mean_num_split_features`: Mean number of split features (lower is better)
- `std_dev_absorption_fraction_score`: Standard deviation of absorption fraction score
- `std_dev_full_absorption_score`: Standard deviation of full absorption score
- `std_dev_num_split_features`: Standard deviation of number of split features

### Unlearning Metrics (`results/unlearning/*.json`)
- When available, metrics will be automatically tracked from this directory

---

For information on how to extract, view, and compare metrics, see the [Metrics Tracking](README.md#metrics-tracking) section in README.md.
