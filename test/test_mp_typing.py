#!/usr/bin/env python3
"""Test script to verify jaxtyping annotations don't break matching_pursuit_sae."""

import torch as t
from sae_research.matching_pursuit_sae import (
    MatchingPursuitAutoEncoder,
    NestedMatchingPursuitAutoEncoder,
    geometric_median,
)

# Test parameters
activation_dim = 64
dict_size = 128
n_tokens = 32
s = 10
s_values = [5, 10, 15]

print("Testing geometric_median...")
points = t.randn(n_tokens, activation_dim)
median = geometric_median(points)
assert median.shape == (activation_dim,), f"Expected shape (activation_dim,), got {median.shape}"
print("✓ geometric_median works")

print("\nTesting MatchingPursuitAutoEncoder...")
ae = MatchingPursuitAutoEncoder(activation_dim, dict_size, s)
x = t.randn(n_tokens, activation_dim)

# Test encode
z = ae.encode(x)
assert z.shape == (n_tokens, dict_size), f"Expected shape ({n_tokens}, {dict_size}), got {z.shape}"
print("✓ encode() works")

# Test encode with return_info
z, indices, initial_acts = ae.encode(x, return_info=True)
assert z.shape == (n_tokens, dict_size), f"Expected z shape ({n_tokens}, {dict_size}), got {z.shape}"
assert indices.shape == (n_tokens, s), f"Expected indices shape ({n_tokens}, {s}), got {indices.shape}"
assert initial_acts.shape == (n_tokens, dict_size), f"Expected initial_acts shape ({n_tokens}, {dict_size}), got {initial_acts.shape}"
print("✓ encode(return_info=True) works")

# Test decode
x_hat = ae.decode(z)
assert x_hat.shape == (n_tokens, activation_dim), f"Expected shape ({n_tokens}, {activation_dim}), got {x_hat.shape}"
print("✓ decode() works")

# Test forward
x_hat = ae.forward(x)
assert x_hat.shape == (n_tokens, activation_dim), f"Expected shape ({n_tokens}, {activation_dim}), got {x_hat.shape}"
print("✓ forward() works")

x_hat, z = ae.forward(x, output_features=True)
assert x_hat.shape == (n_tokens, activation_dim)
assert z.shape == (n_tokens, dict_size)
print("✓ forward(output_features=True) works")

print("\nTesting NestedMatchingPursuitAutoEncoder...")
nested_ae = NestedMatchingPursuitAutoEncoder(activation_dim, dict_size, s_values)

# Test encode
z = nested_ae.encode(x)
assert z.shape == (n_tokens, dict_size), f"Expected shape ({n_tokens}, {dict_size}), got {z.shape}"
print("✓ nested encode() works")

# Test encode_nested
nested_codes = nested_ae.encode_nested(x)
assert isinstance(nested_codes, dict), "encode_nested should return a dict"
assert set(nested_codes.keys()) == set(s_values), f"Expected keys {s_values}, got {list(nested_codes.keys())}"
for s_val, code in nested_codes.items():
    assert code.shape == (n_tokens, dict_size), f"Expected shape ({n_tokens}, {dict_size}) for s={s_val}, got {code.shape}"
print("✓ nested encode_nested() works")

# Test encode_with_info
nested_codes, indices, coeffs, initial_acts = nested_ae.encode_with_info(x)
assert isinstance(nested_codes, dict)
assert indices.shape == (n_tokens, max(s_values))
assert coeffs.shape == (n_tokens, max(s_values))
assert initial_acts.shape == (n_tokens, dict_size)
print("✓ nested encode_with_info() works")

# Test decode
x_hat = nested_ae.decode(z)
assert x_hat.shape == (n_tokens, activation_dim)
print("✓ nested decode() works")

# Test forward
x_hat = nested_ae.forward(x)
assert x_hat.shape == (n_tokens, activation_dim)
print("✓ nested forward() works")

print("\n✅ All typing annotations work correctly!")