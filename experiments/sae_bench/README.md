# SAE Bench DVC Pipeline

A reproducible, modular pipeline for evaluating Sparse Autoencoder (SAE) variants using the [SAE Bench](https://github.com/canrager/sae_bench) framework, managed with [DVC](https://dvc.org).

## Overview

This pipeline evaluates SAE models across multiple metrics:
- **Core metrics**: Reconstruction loss, L0 sparsity, explained variance
- **Sparse probing**: Downstream classification task performance
- **Optional evaluations**: Absorption, SCR, TPP, Unlearning (for larger models)

The pipeline supports custom SAE variants (IHTP, MPSAE) and automatically compares them against baseline implementations.

## Directory Structure

```
experiments/sae_bench/
├── dvc.yaml              # Pipeline definition
├── params.yaml           # Configuration parameters
├── .gitignore           # Ignored files
├── README.md            # This file
└── scripts/             # Pipeline stage scripts
    ├── setup_saes.py
    ├── run_core_eval.py
    ├── run_sparse_probing.py
    ├── run_optional_evals.py
    ├── process_results.py
    └── visualize_results.py
```

## Prerequisites

1. **Install dependencies**:
   ```bash
   pip install dvc pyyaml torch
   pip install sae-bench  # or install from source
   ```

2. **Install custom SAE implementations**:
   Make sure `sae_research/instance_sae.py` is available in the project root.

3. **Initialize DVC** (if not already done):
   ```bash
   cd /path/to/sae_research
   dvc init
   ```

## Configuration

All configuration is in `params.yaml`. Key sections:

### Model Configuration
```yaml
model:
  name: "pythia-70m-deduped"
  llm_batch_size: 512
  torch_dtype: "float32"
  device: "cuda"  # or "cpu"
```

### SAE Variants
Enable/disable and configure IHTP and MPSAE variants:
```yaml
sae_variants:
  ihtp:
    enabled: true
    k_values: [5, 10]  # Top-k sparsity
  mpsae:
    enabled: true
    s_values: [50, 100]  # Number of pursuit steps
```

### Evaluation Types
Toggle which evaluations to run:
```yaml
eval_types:
  core: true
  sparse_probing: true
  absorption: false  # Not recommended for models < 2B params
  scr: false
  tpp: false
  unlearning: false
```

### Baseline SAE
Configure which baseline SAE to load from HuggingFace:
```yaml
baseline_sae:
  repo_id: "canrager/lm_sae"
  filename: "pythia70m_sweep_standard_ctx128_0712/resid_post_layer_4/trainer_8/ae.pt"
  hook_layer: 4
```

## Running the Pipeline

### Run entire pipeline
```bash
cd experiments/sae_bench
dvc repro
```

### Run specific stage
```bash
# Run only setup
dvc repro setup

# Run a specific optional evaluation
dvc repro scr_eval

# Run all optional evaluations
dvc repro scr_eval tpp_eval absorption_eval unlearning_eval

# Run up to visualization
dvc repro visualize
```

**Benefits of split stages:**
- Clear visibility: See exactly which evaluation is running
- Granular control: Run only the evaluations you need
- Better error handling: If one evaluation fails, others can continue
- Progress tracking: Check `dvc.lock` to see which evaluations completed

### View pipeline DAG
```bash
dvc dag
```

Example output:
```
         +-------+
         | setup |
         +-------+
              *
              *
              *
    *********************
    **                 **
    **                 **
+----------+      +---------------------+
|core_eval |      |sparse_probing_eval  |
+----------+      +---------------------+
    **                 **
    **                 **
    *********************
              *
              *
              *
   +-----------------+
   |process_results  |
   +-----------------+
              *
              *
              *
       +-----------+
       | visualize |
       +-----------+
```

### Run individual scripts (without DVC)
You can also run scripts directly for development/debugging:
```bash
cd experiments/sae_bench
python scripts/setup_saes.py
python scripts/run_core_eval.py
```

## Pipeline Stages

### 1. Setup (`setup`)
- Loads baseline SAE from HuggingFace
- Creates custom variants (IHTP with k=5,10; MPSAE with s=50,100)
- Loads comparison SAEs from SAE Bench
- **Outputs**: `downloaded_saes/` (SAE weights and configs)

### 2. Core Evaluation (`core_eval`)
- Runs basic reconstruction and sparsity metrics
- Required for all downstream stages
- **Outputs**: `eval_results/core/`

### 3. Sparse Probing (`sparse_probing_eval`)
- Tests SAE features on classification tasks
- Uses datasets like "LabHC/bias_in_bios_class_set1"
- **Outputs**: `eval_results/sparse_probing/`

### 4. Optional Evaluations (Split into Separate Stages)
Each optional evaluation runs as an independent stage for better visibility:

#### 4a. SCR Evaluation (`scr_eval`)
- Spurious Correlation Removal evaluation
- **Outputs**: `results/scr/`

#### 4b. TPP Evaluation (`tpp_eval`)
- Targeted Probe Perturbation evaluation
- **Outputs**: `results/tpp/`

#### 4c. Absorption Evaluation (`absorption_eval`)
- Feature absorption detection
- **Outputs**: `results/absorption/`

#### 4d. Unlearning Evaluation (`unlearning_eval`)
- Selective knowledge removal
- **Outputs**: `results/unlearning/`

### 5. Process Results (`process_results`)
- Merges core metrics with eval-specific metrics
- **Outputs**: `eval_results/merged/`

### 6. Visualize (`visualize`)
- Generates L0 sparsity vs. metric comparison plots
- Different shapes/colors for SAE architectures
- **Outputs**: `images/*.png`

## Key Metrics

- **L0 Sparsity**: Average number of active features per token
- **Reconstruction Loss**: MSE between input and reconstructed activations
- **Loss Recovered**: % of model loss recovered after SAE intervention
- **Probe Accuracy**: Top-k accuracy on downstream classification tasks

## Customization

### Add new SAE variant
1. Implement variant in `sae_research/instance_sae.py`
2. Update `scripts/setup_saes.py` to create variant
3. Add configuration to `params.yaml`
4. Add marker/color to visualization settings

### Add new evaluation dataset
1. Update `sparse_probing.dataset_names` in `params.yaml`
2. Run pipeline: `dvc repro`

### Change baseline SAE
1. Update `baseline_sae` section in `params.yaml`
2. Delete cached SAEs: `rm -rf downloaded_saes/`
3. Re-run pipeline: `dvc repro`

## Performance Notes

- **Runtime**: Core + sparse probing ~2 minutes on RTX 3090
- **Memory**: Adjust `llm_batch_size` based on GPU memory
- **Disk space**: Set `save_activations: true` requires ~100GB for caching
- **Optional evals**: Full evaluation suite ~1 hour per layer

## Outputs

All outputs are relative to project root:

- `downloaded_saes/`: SAE weights and configs (DVC-tracked)
- `eval_results/core/`: Core evaluation results
- `eval_results/sparse_probing/`: Sparse probing results
- `eval_results/merged/`: Merged results (JSON)
- `images/`: Comparison plots (PNG)

## Troubleshooting

### Import errors
Make sure project root is in PYTHONPATH:
```bash
export PYTHONPATH=/path/to/sae_research:$PYTHONPATH
```

### CUDA out of memory
Reduce batch size in `params.yaml`:
```yaml
model:
  llm_batch_size: 256  # or lower
```

### Missing SAE Bench
Install from source:
```bash
git clone https://github.com/canrager/sae_bench
cd sae_bench
pip install -e .
```

## References

- SAE Bench: https://github.com/canrager/sae_bench
- DVC Documentation: https://dvc.org/doc
- Pipeline design based on: `experiments/PIPELINES.md`
- Original implementation: `sae_research/sae_bench_demo.py`

## Metrics Tracking

### Metrics Extraction Stages

The pipeline includes dedicated stages for extracting metrics from evaluation results:

- **`core_metrics`**: Extracts metrics from core evaluation
- **`sparse_probing_metrics`**: Extracts metrics from sparse probing
- **`scr_metrics`**: Extracts metrics from SCR evaluation
- **`tpp_metrics`**: Extracts metrics from TPP evaluation
- **`absorption_metrics`**: Extracts metrics from absorption evaluation
- **`unlearning_metrics`**: Extracts metrics from unlearning evaluation
- **`aggregate_metrics`**: Combines all metrics into a single file

### Running Metrics Extraction

```bash
# Extract metrics without re-running evaluations
uv run dvc repro core_metrics sparse_probing_metrics scr_metrics tpp_metrics

# Or extract all metrics at once
uv run dvc repro aggregate_metrics

# Extract metrics for specific evaluation type
uv run dvc repro core_metrics
```

### View Current Metrics

```bash
# Show all metrics from the latest run
uv run dvc metrics show

# Show metrics in JSON format
uv run dvc metrics show --json

# Show metrics from specific metric files
uv run dvc metrics show metrics/core.json
uv run dvc metrics show metrics/sparse_probing.json

# Show aggregated metrics
uv run dvc metrics show metrics/all_metrics.json
```

### Compare Metrics Across Experiments

```bash
# Compare metrics between workspace and a specific commit/tag
uv run dvc metrics diff <commit-hash>

# Compare metrics between two commits
uv run dvc metrics diff <commit1> <commit2>
```

### Track Metrics Across Experiments

```bash
# Run an experiment
uv run dvc exp run

# View metrics from all experiments
uv run dvc exp show

# Compare specific experiments
uv run dvc exp diff exp-1 exp-2
```

### Visualizing Metrics

```bash
# Generate plots for tracked metrics (if configured)
uv run dvc plots show

# Generate plots comparing experiments
uv run dvc plots diff <commit1> <commit2>
```

### How Metrics Extraction Works

1. Each evaluation stage produces JSON files in its output directory
2. Metrics extraction stages run `scripts/extract_metrics.py` to parse evaluation results
3. The extraction script:
   - Reads all JSON files from the specified results directory
   - Extracts key metrics based on the evaluation type
   - Saves metrics to lightweight JSON files in the `metrics/` directory
4. DVC tracks these metric files with `cache: false` to always read fresh values
5. Metrics are tracked in Git alongside code changes

### The Unified Extraction Script

The `scripts/extract_metrics.py` script handles all metric extraction:

```bash
# Extract specific metric type
python scripts/extract_metrics.py core results/core metrics/core.json

# Aggregate multiple metric files
python scripts/extract_metrics.py aggregate metrics/core.json metrics/sparse_probing.json metrics/all.json
```

Supported metric types: `core`, `sparse_probing`, `scr`, `tpp`, `absorption`, `unlearning`

### Adding New Metrics

To track additional metrics:

1. Modify the appropriate extraction function in `scripts/extract_metrics.py`:
   - For core metrics: Edit `extract_core_metrics()`
   - For sparse probing: Edit `extract_sparse_probing_metrics()`
   - For other types: Edit the corresponding function

2. The extraction functions read from the `eval_result_metrics` field in the evaluation JSON files

3. Re-run the metrics extraction stage:
   ```bash
   uv run dvc repro core_metrics  # or the appropriate metrics stage
   ```

4. The new metrics will automatically be tracked by DVC

Example of adding a new metric to core extraction:
```python
def extract_core_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    metrics = {}
    result_metrics = data.get('eval_result_metrics', {})

    # Add your new metric extraction
    if 'your_metric_category' in result_metrics:
        metrics['your_metric_name'] = result_metrics['your_metric_category'].get('your_metric_key')

    return metrics
```

### Metrics Tracking Notes

- Metrics extraction stages are independent of evaluation stages, allowing you to:
  - Re-extract metrics without re-running evaluations
  - Modify metric selection without touching evaluation code
  - Add new metrics retroactively from existing results
- Metrics files are not cached by DVC (due to `cache: false`), so they're always read fresh
- Metrics are stored in lightweight JSON files, making them Git-friendly
- Multiple SAE evaluations in the same directory will all be tracked
- The aggregated metrics file (`metrics/all_metrics.json`) provides a unified view of all metrics

## See Also

- `METRICS.md` - Tracked metrics definitions and latest evaluation results
- `experiments/PIPELINES.md` - Detailed pipeline documentation
- `sae_research/instance_sae.py` - Custom SAE implementations
