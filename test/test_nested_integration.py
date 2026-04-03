#!/usr/bin/env python3
"""Test script to verify nested thresholding SAE integration with demo_config."""

from sae_research.training.config import get_trainer_configs, TrainerType

# Test parameters
architectures = [TrainerType.NESTED_THRESHOLDING_TOP_K.value]
learning_rates = [5e-5]
seeds = [0]
activation_dim = 768
dict_sizes = [2**14]
model_name = "test_model"
device = "cuda"
layer = "blocks.10.mlp.hook_post"
submodule_name = "test_submodule"
steps = 10000

# Get trainer configs
configs = get_trainer_configs(
    architectures=architectures,
    learning_rates=learning_rates,
    seeds=seeds,
    activation_dim=activation_dim,
    dict_sizes=dict_sizes,
    model_name=model_name,
    device=device,
    layer=layer,
    submodule_name=submodule_name,
    steps=steps,
)

print(f"Generated {len(configs)} configurations")

if configs:
    print("\nFirst configuration:")
    for key, value in configs[0].items():
        print(f"  {key}: {value}")

    # Check that the nested-specific fields are present
    assert "k_values" in configs[0], "k_values field missing"
    assert "k_weights" in configs[0], "k_weights field missing"
    assert isinstance(configs[0]["k_values"], list), "k_values should be a list"
    assert isinstance(configs[0]["k_weights"], list), "k_weights should be a list"

    print("\nk_values:", configs[0]["k_values"])
    print("k_weights:", configs[0]["k_weights"])

    print("\n✓ Integration test passed!")
else:
    print("ERROR: No configurations generated")
