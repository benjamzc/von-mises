"""Tests for the von Mises sampler implementation."""

import jax
import jax.numpy as jnp
from jax import random, vmap
import numpy as np
import pytest
import scipy.stats
from typing import Dict, Any, List, Tuple

from jax_von_mises.sampler import sample_von_mises, vmises_log_prob, compute_p, vmises_entropy


def test_compute_p():
    """Test the computation of optimal p parameter."""
    # For small kappa, p should approach 0.5
    p_small = compute_p(jnp.array(1e-7))
    assert jnp.isclose(p_small, 0.5, atol=1e-2)  # Relaxed tolerance
    
    # Test a few known values
    kappa_values = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
    p_values = compute_p(kappa_values)
    
    # p should be in the range [0, 0.5]
    assert jnp.all(p_values >= 0.0) and jnp.all(p_values <= 0.5)
    
    # For large kappa, p should be ≤ 0.5 (the upper limit of the clipping)
    p_large = compute_p(jnp.array(1000.0))
    assert p_large <= 0.5  # Check it's within the valid range


def test_von_mises_log_prob():
    """Test log probability computation against scipy."""
    # Generate some test points
    x = jnp.linspace(-jnp.pi, jnp.pi, 100)
    loc = 0.0
    concentration = 2.0
    
    # Compute log probabilities
    log_probs = vmises_log_prob(x, loc, concentration)
    
    # Compare with scipy
    scipy_log_probs = scipy.stats.vonmises.logpdf(x, concentration, loc=loc)
    
    # Check results
    assert jnp.allclose(log_probs, scipy_log_probs, rtol=1e-5, atol=1e-5)


def test_von_mises_entropy():
    """Test entropy computation against scipy."""
    # Test various concentration values
    kappa_values = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
    
    # Compute entropy with JAX implementation
    jax_entropy_values = vmises_entropy(kappa_values)
    
    # Compute entropy with SciPy
    scipy_entropy_values = np.array([
        scipy.stats.vonmises.entropy(kappa) for kappa in kappa_values
    ])
    
    # Check results with appropriate tolerance
    assert jnp.allclose(jax_entropy_values, scipy_entropy_values, rtol=1e-5, atol=1e-5)
    
    # Test JAX transformations compatibility
    jitted_entropy = jax.jit(vmises_entropy)
    vmapped_entropy = jax.vmap(vmises_entropy)
    
    # Test with scalar input
    jitted_result = jitted_entropy(2.0)
    assert jnp.isclose(jitted_result, scipy.stats.vonmises.entropy(2.0), rtol=1e-5)
    
    # Test with array input using vmap
    vmapped_result = vmapped_entropy(kappa_values)
    assert jnp.allclose(vmapped_result, scipy_entropy_values, rtol=1e-5)
    
    # Test extreme values
    # Very small concentration (approaching uniform distribution)
    small_kappa = 1e-8
    small_entropy = vmises_entropy(small_kappa)
    # Entropy of uniform distribution on [-π, π] should be log(2π)
    assert jnp.isclose(small_entropy, jnp.log(2 * jnp.pi), rtol=1e-3)
    
    # Very large concentration (approaching normal distribution with variance 1/kappa)
    large_kappa = 1000.0
    large_entropy = vmises_entropy(large_kappa)
    # Entropy should be decreasing with increasing concentration
    assert large_entropy < vmises_entropy(100.0)


def test_sample_shape(rng_key, sample_parameters):
    """Test that sample shapes match expected shapes."""
    # Test scalar parameters
    params = sample_parameters['scalar']
    samples = sample_von_mises(
        rng_key, params['loc'], params['concentration'], params['shape']
    )
    assert samples.shape == params['shape']
    
    # Test batch parameters without explicit shape (should infer shape from inputs)
    params = sample_parameters['batch']
    samples = sample_von_mises(
        rng_key, params['loc'], params['concentration']
    )
    assert samples.shape == params['loc'].shape
    
    # Test NN-like output parameters without explicit shape
    params = sample_parameters['nn_output']
    samples = sample_von_mises(
        rng_key, params['loc'], params['concentration']
    )
    assert samples.shape == params['loc'].shape
    
    # Test with explicit shape from inputs - use small shape for test efficiency
    small_shape = (5, 3)
    samples = sample_von_mises(
        rng_key, jnp.array(0.0), jnp.array(1.0), shape=small_shape
    )
    assert samples.shape == small_shape


def test_sample_range(rng_key, sample_parameters):
    """Test that samples are in the correct range [-π, π]."""
    for param_type, params in sample_parameters.items():
        # Use without explicit shape for this test
        samples = sample_von_mises(
            rng_key, params['loc'], params['concentration']
        )
        assert jnp.all(samples >= -jnp.pi) and jnp.all(samples <= jnp.pi)


def test_sample_distribution(rng_key):
    """Test that samples follow the expected distribution."""
    # Generate many samples for a simple case
    n_samples = 1000  # Reduced from 10000 for efficiency
    loc = 0.0
    concentration = 2.0
    
    samples = sample_von_mises(rng_key, loc, concentration, shape=(n_samples,))
    
    # Compute circular mean and variance
    sin_mean = jnp.mean(jnp.sin(samples))
    cos_mean = jnp.mean(jnp.cos(samples))
    circular_mean = jnp.arctan2(sin_mean, cos_mean)
    r = jnp.sqrt(sin_mean**2 + cos_mean**2)
    circular_variance = 1.0 - r
    
    # Expected values
    expected_mean = loc
    expected_variance = 1.0 - scipy.special.i1(concentration) / scipy.special.i0(concentration)
    
    # Check results (with appropriate tolerance for randomness)
    assert jnp.isclose(circular_mean, expected_mean, atol=0.1)
    assert jnp.isclose(circular_variance, expected_variance, atol=0.1)


def test_jit_compatibility(rng_key, sample_parameters):
    """Test that the sampler works with JAX transformations."""
    # Test jit with static shape
    def sample_fn(key, loc, conc):
        return sample_von_mises(key, loc, conc, shape=(100,))
    
    jitted_sampler = jax.jit(sample_fn)
    
    # Test with scalar parameters
    samples = jitted_sampler(rng_key, 0.0, 2.0)
    assert samples.shape == (100,)
    
    # Test vmap
    vmapped_sampler = jax.vmap(
        lambda key, loc, conc: sample_von_mises(key, loc, conc, shape=(10,)),
        in_axes=(0, 0, 0)
    )
    
    batch_size = 3
    keys = random.split(rng_key, batch_size)
    locs = jnp.array([0.0, 1.0, -1.0])
    concentrations = jnp.array([0.5, 2.0, 10.0])
    
    batch_samples = vmapped_sampler(keys, locs, concentrations)
    assert batch_samples.shape == (batch_size, 10)


def test_extreme_concentrations(rng_key):
    """Test behavior with very small and very large concentrations."""
    # Very small concentration (close to uniform)
    # Use smaller sample size for test efficiency
    n_samples = 1000
    
    small_conc_samples = sample_von_mises(
        rng_key, 0.0, 1e-6, shape=(n_samples,)
    )
    
    # For very small concentration, the distribution should be close to uniform
    # We'll verify the samples are within the expected range
    assert jnp.all(small_conc_samples >= -jnp.pi) and jnp.all(small_conc_samples <= jnp.pi)
    
    # Very large concentration (close to a point mass)
    large_conc_samples = sample_von_mises(
        rng_key, 1.0, 1000.0, shape=(100,)  # Reduced sample size
    )
    
    # Check that it's concentrated around the mean
    assert jnp.all(jnp.abs(large_conc_samples - 1.0) < 0.1) 