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

# Run up to visualization
dvc repro visualize
```

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

### 4. Optional Evaluations (`optional_evals`)
- Runs absorption, SCR, TPP, unlearning if enabled
- **Outputs**: `eval_results/{absorption,scr,tpp,unlearning}/`

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

## See Also

- `experiments/PIPELINES.md` - Detailed pipeline documentation
- `sae_research/instance_sae.py` - Custom SAE implementations
