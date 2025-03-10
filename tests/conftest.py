"""Test configuration and fixtures."""

import jax
import jax.numpy as jnp
from jax import random, vmap
import numpy as np
import pytest
from typing import Dict, Any, List, Tuple

@pytest.fixture
def rng_key():
    """Return a fixed PRNG key for reproducible tests."""
    return random.PRNGKey(42)

@pytest.fixture
def sample_parameters():
    """Return sample parameters for testing."""
    return {
        'scalar': {
            'loc': 0.0, 
            'concentration': 2.0,
            'shape': (1000,)
        },
        'batch': {
            'loc': jnp.array([0.0, 1.0, -1.0]),
            'concentration': jnp.array([0.5, 2.0, 10.0]),
            'shape': (1000, 3)
        },
        'nn_output': {
            'loc': jnp.array([[0.0, 1.0], [-1.0, 0.5]]),
            'concentration': jnp.array([[0.5, 2.0], [5.0, 10.0]]),
            'shape': (1000, 2, 2)
        }
    }

@pytest.fixture
def mock_neural_network():
    """Return a mock neural network that outputs von Mises parameters."""
    def init_params(key, input_shape):
        k1, k2 = random.split(key)
        scale = 0.1
        return {
            'loc': {
                'w': scale * random.normal(k1, (input_shape[-1], 1)),
                'b': jnp.zeros(1),
            },
            'concentration': {
                'w': scale * random.normal(k2, (input_shape[-1], 1)),
                'b': jnp.ones(1),
            }
        }
    
    def apply_fn(params, x):
        loc = jnp.dot(x, params['loc']['w']) + params['loc']['b']
        conc = jnp.exp(jnp.dot(x, params['concentration']['w']) + params['concentration']['b'])
        return loc, conc
    
    return {'init_params': init_params, 'apply_fn': apply_fn} 