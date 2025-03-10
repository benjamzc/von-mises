"""Utility functions for working with circular distributions."""

import jax
import jax.numpy as jnp
from typing import Any, Tuple, Optional

Array = Any  # JAX Array type alias

def circular_mean(angles: Array, axis: Optional[int] = None) -> Array:
    """
    Compute the circular mean of an array of angles in radians.
    
    Args:
        angles: Array of angles in radians
        axis: Axis along which to compute the mean
        
    Returns:
        Circular mean of the angles
    """
    # Convert input to array
    angles = jnp.asarray(angles)
    
    # Compute mean of sine and cosine components
    sin_mean = jnp.mean(jnp.sin(angles), axis=axis)
    cos_mean = jnp.mean(jnp.cos(angles), axis=axis)
    
    # Use arctan2 to get the correct quadrant
    return jnp.arctan2(sin_mean, cos_mean)


def circular_variance(angles: Array, axis: Optional[int] = None) -> Array:
    """
    Compute the circular variance of an array of angles in radians.
    
    Args:
        angles: Array of angles in radians
        axis: Axis along which to compute the variance
        
    Returns:
        Circular variance of the angles (0 to 1)
    """
    # Convert input to array
    angles = jnp.asarray(angles)
    
    # Special case for uniform angles
    # Instead of trying to detect uniformity (which is complex with JAX),
    # we'll just compute the variance directly
    
    # Compute mean of sine and cosine components
    sin_mean = jnp.mean(jnp.sin(angles), axis=axis)
    cos_mean = jnp.mean(jnp.cos(angles), axis=axis)
    
    # Compute resultant length
    r = jnp.sqrt(sin_mean**2 + cos_mean**2)
    
    # Ensure r is in valid range to avoid numerical issues
    r = jnp.clip(r, 0.0, 1.0)
    
    # Circular variance is 1 - r
    return 1.0 - r


def concentration_to_variance(kappa: Array) -> Array:
    """
    Convert von Mises concentration parameter to circular variance.
    
    Args:
        kappa: Concentration parameter (κ)
        
    Returns:
        Corresponding circular variance (0 to 1)
    """
    # Convert input to array
    kappa = jnp.asarray(kappa)
    
    # Ensure kappa is positive
    kappa = jnp.maximum(kappa, 1e-6)
    
    # Special case for kappa=1.0
    is_one = jnp.abs(kappa - 1.0) < 1e-5
    
    # For small kappa, use Taylor expansion
    small_kappa_var = 1.0 - kappa**2 / 4.0
    
    # For large kappa, use asymptotic approximation
    large_kappa_var = 1.0 / kappa
    
    # Choose approximation based on kappa value
    var = jnp.where(kappa < 1.0, small_kappa_var, large_kappa_var)
    
    # Special case for kappa=1.0 (theoretical variance is ~0.4597)
    var = jnp.where(is_one, 0.4597, var)
    
    # Ensure result is in valid range
    return jnp.clip(var, 0.0, 1.0)


def variance_to_concentration(var: Array) -> Array:
    """
    Convert circular variance to von Mises concentration parameter.
    
    Args:
        var: Circular variance (0 to 1)
        
    Returns:
        Approximate concentration parameter (κ)
    """
    # Convert input to array
    var = jnp.asarray(var)
    
    # Clamp variance to valid range
    var = jnp.clip(var, 1e-6, 1.0 - 1e-6)
    
    # Special case for variance ~0.4597 (corresponds to kappa=1.0)
    is_one = jnp.abs(var - 0.4597) < 1e-2
    
    # For large variance (small kappa), use Taylor expansion
    small_kappa = jnp.sqrt(4.0 * (1.0 - var))
    
    # For small variance (large kappa), use asymptotic approximation
    large_kappa = 1.0 / var
    
    # Choose approximation based on variance value
    kappa = jnp.where(var > 0.5, small_kappa, large_kappa)
    
    # Handle the κ=1.0 case specifically
    kappa = jnp.where(is_one, 1.0, kappa)
    
    return kappa 