"""Utilities for integrating von Mises sampling with neural networks."""

import jax
import jax.numpy as jnp
from jax import random
from typing import Any, Callable, Dict, Tuple, Union, Optional

from jax_von_mises.sampler import sample_von_mises

Array = Any  # JAX Array type alias
ModelApplyFn = Callable[[Dict[str, Any], Array], Tuple[Array, Array]]

def von_mises_layer(
    key: Array,
    mean_logits: Array,
    concentration: Array,
    temperature: float = 1.0,
    training: bool = True,
) -> Tuple[Array, Array]:
    """
    Sample from a von Mises distribution for a directional layer in a neural network.
    
    Args:
        key: PRNG key
        mean_logits: Unnormalized predicted angles (logits)
        concentration: Concentration parameter κ > 0
        temperature: Temperature for sampling (higher = more diversity)
        training: Whether to sample (True) or return the mean (False)
        
    Returns:
        Tuple of (samples, mean); samples are used during training, mean during inference
    """
    # Convert inputs to arrays
    mean_logits = jnp.asarray(mean_logits)
    concentration = jnp.asarray(concentration)
    
    # Normalize mean to [-π, π] using atan2 to handle the periodicity
    mean = jnp.arctan2(jnp.sin(mean_logits), jnp.cos(mean_logits))
    
    # Scale concentration by temperature
    scaled_concentration = concentration / jnp.maximum(temperature, 1e-6)
    
    # Sample or return mean based on training flag
    samples = jax.lax.cond(
        training,
        lambda: sample_von_mises(key, mean, scaled_concentration),
        lambda: mean
    )
    
    return samples, mean


def pmap_compatible_von_mises_sampling(
    model_apply_fn: ModelApplyFn,
    params: Dict[str, Any],
    inputs: Array,
    rng_key: Array,
    temperature: float = 1.0,
    training: bool = True,
) -> Array:
    """
    Wrapper for using von Mises sampling within jax.pmap context.
    
    This function is designed to be used with neural networks that output
    parameters for von Mises distributions.
    
    Args:
        model_apply_fn: Function to apply model parameters to inputs
        params: Model parameters
        inputs: Model inputs
        rng_key: PRNG key
        temperature: Temperature for sampling
        training: Whether to sample or return mean
        
    Returns:
        Directional samples or means from the model output
    """
    # Call model to get distribution parameters
    mean_logits, concentration = model_apply_fn(params, inputs)
    
    # Ensure positive concentration
    concentration = jnp.maximum(concentration, 1e-6)
    
    # Sample from von Mises
    samples, mean = von_mises_layer(
        rng_key, 
        mean_logits, 
        concentration, 
        temperature=temperature,
        training=training
    )
    
    return samples 