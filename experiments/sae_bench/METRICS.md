# SAEBench Evaluation Metrics

## Overview

This document defines the metrics tracked by the SAEBench evaluation pipeline and presents the latest evaluation results.

## Latest Evaluation Results

### Evaluation Configuration
- **Model**: google/gemma-2-2b
- **Layer**: 11 (resid_post)
- **Dictionary size**: 4608 (2x expansion)
- **Context length**: 128
- **Training tokens**: 200,000,000
- **SAE Variants Tested**: 6

### Core Metrics

| SAE Variant                                                  |   KL Div |   CE Loss |   CE Score |   Explained Var |   MSE |   CosSim |    L0 |   L1 |   L2 Ratio |
|:-------------------------------------------------------------|---------:|----------:|-----------:|----------------:|------:|---------:|------:|-----:|-----------:|
| gemma-scope-2b-pt-res-canonical_layer_11_width_16k_canonical |    0.99  |      3.09 |      0.99  |           0.875 |   5.9 |    0.926 |  78.7 |  524 |      0.922 |
| gemma-scope-2b-pt-res-canonical_layer_11_width_65k_canonical |    0.992 |      3.08 |      0.992 |           0.895 |   5.1 |    0.934 |  69.4 |  462 |      0.93  |
| gemma22b_nested_topk_k40_layer11_trainer0_custom_sae         |    0.931 |      3.67 |      0.929 |           0.758 |  11.6 |    0.848 |  40   |  -18 |      0.852 |
| gemma22b_nested_topk_k80_layer11_trainer0_custom_sae         |    0.963 |      3.36 |      0.962 |           0.797 |   9.8 |    0.875 |  80   |  -18 |      0.887 |
| gemma22b_nested_topk_k160_layer11_trainer0_custom_sae        |    0.984 |      3.14 |      0.985 |           0.832 |   8   |    0.898 | 160   |  -20 |      0.945 |
| gemma22b_nested_topk_k320_layer11_trainer0_custom_sae        |    0.993 |      3.06 |      0.993 |           0.836 |   8   |    0.91  | 320   |  -22 |      1.047 |

**Key Observations**:
- [Manual interpretation goes here after generation]

### Sparse Probing Results

| SAE Variant                                                  |   Test Accuracy |   Top-1 |   Top-2 |   Top-5 |
|:-------------------------------------------------------------|----------------:|--------:|--------:|--------:|
| LLM Baseline                                                 |           0.967 |   0.663 |   0.723 |   0.792 |
| gemma-scope-2b-pt-res-canonical_layer_11_width_16k_canonical |           0.96  |   0.868 |   0.87  |   0.901 |
| gemma-scope-2b-pt-res-canonical_layer_11_width_65k_canonical |           0.959 |   0.786 |   0.833 |   0.898 |
| gemma22b_nested_topk_k40_layer11_trainer0_custom_sae         |           0.95  |   0.789 |   0.805 |   0.862 |
| gemma22b_nested_topk_k80_layer11_trainer0_custom_sae         |           0.958 |   0.785 |   0.806 |   0.863 |
| gemma22b_nested_topk_k160_layer11_trainer0_custom_sae        |           0.96  |   0.791 |   0.826 |   0.878 |
| gemma22b_nested_topk_k320_layer11_trainer0_custom_sae        |           0.963 |   0.779 |   0.821 |   0.883 |

**Key Observations**:
- [Manual interpretation goes here after generation]

**Sparsity vs Interpretability Trade-off**:

![L0 vs Sparse Probing Top-1 Accuracy by SAE Type](results/images/sparse_probing_2var_sae_type.png)

- [Manual plot interpretation goes here after generation]

### SCR & TPP Metrics

| SAE Variant                                                  |   SCR (thresh=10) |   TPP Total (thresh=10) |   TPP Intended |   TPP Unintended |
|:-------------------------------------------------------------|------------------:|------------------------:|---------------:|-----------------:|
| gemma-scope-2b-pt-res-canonical_layer_11_width_16k_canonical |             0.258 |                   0.033 |          0.037 |            0.003 |
| gemma-scope-2b-pt-res-canonical_layer_11_width_65k_canonical |             0.175 |                   0.016 |          0.018 |            0.002 |
| gemma22b_nested_topk_k40_layer11_trainer0_custom_sae         |             0.184 |                  -0.001 |          0.006 |            0.007 |
| gemma22b_nested_topk_k80_layer11_trainer0_custom_sae         |             0.202 |                   0.002 |          0.01  |            0.008 |
| gemma22b_nested_topk_k160_layer11_trainer0_custom_sae        |             0.223 |                   0.004 |          0.013 |            0.009 |
| gemma22b_nested_topk_k320_layer11_trainer0_custom_sae        |             0.228 |                   0.006 |          0.015 |            0.009 |

**Key Observations**:
- [Manual interpretation goes here after generation]

### Absorption Metrics

| SAE Variant                                                  |   Absorption Fraction |   Full Absorption |   Split Features |   Absorption Fraction Std |   Full Absorption Std |
|:-------------------------------------------------------------|----------------------:|------------------:|-----------------:|--------------------------:|----------------------:|
| gemma-scope-2b-pt-res-canonical_layer_11_width_16k_canonical |                 0.109 |             0.094 |              1.1 |                     0.123 |                 0.109 |
| gemma-scope-2b-pt-res-canonical_layer_11_width_65k_canonical |                 0.35  |             0.321 |              1.3 |                     0.215 |                 0.211 |
| gemma22b_nested_topk_k40_layer11_trainer0_custom_sae         |                 0.059 |             0.078 |              1.2 |                     0.109 |                 0.106 |
| gemma22b_nested_topk_k80_layer11_trainer0_custom_sae         |                 0.087 |             0.068 |              1.3 |                     0.153 |                 0.094 |
| gemma22b_nested_topk_k160_layer11_trainer0_custom_sae        |                 0.096 |             0.056 |              1.2 |                     0.164 |                 0.084 |
| gemma22b_nested_topk_k320_layer11_trainer0_custom_sae        |                 0.094 |             0.041 |              1.2 |                     0.154 |                 0.062 |

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
