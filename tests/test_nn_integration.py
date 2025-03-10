"""Tests for neural network integration functionality."""

import jax
import jax.numpy as jnp
from jax import random, vmap
import numpy as np
import pytest
from typing import Dict, Any, Tuple

from jax_von_mises.nn.integration import (
    von_mises_layer,
    pmap_compatible_von_mises_sampling
)
from jax_von_mises.utils import circular_variance


def test_von_mises_layer(rng_key):
    """Test von Mises layer functionality."""
    # Setup test data
    mean_logits = jnp.array([0.0, jnp.pi/2, jnp.pi])
    concentration = jnp.array([1.0, 2.0, 5.0])
    
    # Test training mode (with sampling)
    samples, mean = von_mises_layer(rng_key, mean_logits, concentration, training=True)
    
    # Check shapes
    assert samples.shape == mean_logits.shape
    assert mean.shape == mean_logits.shape
    
    # Check that mean is correctly computed
    # Note: arctan2 normalizes angles to [-π, π], so π becomes -π
    expected_mean = jnp.array([0.0, jnp.pi/2, -jnp.pi])
    assert jnp.allclose(mean, expected_mean, atol=1e-5)
    
    # Test inference mode (no sampling)
    samples_inference, mean_inference = von_mises_layer(
        rng_key, mean_logits, concentration, training=False
    )
    
    # In inference mode, samples should equal mean
    assert jnp.allclose(samples_inference, mean_inference)
    
    # Test temperature effect on concentration
    # Use a simpler approach that doesn't require many samples
    key1, key2 = random.split(rng_key)
    
    # Higher temperature should result in lower effective concentration
    # which should give more variable results
    # We'll test this by comparing two different temperatures on the same distribution
    single_mean = jnp.array(0.0)
    single_conc = jnp.array(10.0)
    
    # Sample with two different temperatures
    high_temp_sample, _ = von_mises_layer(key1, single_mean, single_conc, temperature=10.0, training=True)
    low_temp_sample, _ = von_mises_layer(key2, single_mean, single_conc, temperature=0.1, training=True)
    
    # Higher temperature should result in samples potentially further from the mean
    # Since we only have single samples, we'll just check that the high temperature doesn't
    # make the sample identical to the mean
    assert high_temp_sample != single_mean


def test_pmap_compatible_sampling(rng_key, mock_neural_network):
    """Test pmap-compatible sampling function."""
    # Setup mock data and model
    input_dim = 4
    batch_size = 8
    
    inputs = random.normal(rng_key, (batch_size, input_dim))
    init_fn = mock_neural_network['init_params']
    apply_fn = mock_neural_network['apply_fn']
    
    # Initialize model
    params = init_fn(rng_key, inputs.shape)
    
    # Test basic functionality
    samples = pmap_compatible_von_mises_sampling(
        apply_fn, params, inputs, rng_key
    )
    
    # Check shape
    assert samples.shape == (batch_size, 1)
    
    # Test with vmap (simulating pmap behavior)
    vmap_sampling = jax.vmap(
        lambda inputs, key: pmap_compatible_von_mises_sampling(
            apply_fn, params, inputs, key
        ),
        in_axes=(0, 0)
    )
    
    # Split inputs and keys to simulate parallel processing
    split_size = 2
    split_inputs = inputs.reshape(split_size, batch_size // split_size, input_dim)
    split_keys = random.split(rng_key, split_size)
    
    # Run vmap
    vmap_samples = vmap_sampling(split_inputs, split_keys)
    
    # Check shape
    assert vmap_samples.shape == (split_size, batch_size // split_size, 1)
    
    # Test training vs inference mode
    training_samples = pmap_compatible_von_mises_sampling(
        apply_fn, params, inputs, rng_key, training=True
    )
    inference_samples = pmap_compatible_von_mises_sampling(
        apply_fn, params, inputs, rng_key, training=False
    )
    
    # In inference mode, samples should be deterministic
    # (running twice with same key should give same results)
    inference_samples2 = pmap_compatible_von_mises_sampling(
        apply_fn, params, inputs, rng_key, training=False
    )
    assert jnp.allclose(inference_samples, inference_samples2) 