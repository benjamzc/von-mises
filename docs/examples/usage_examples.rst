Usage Examples
==============

This document provides practical examples of how to use the JAX von Mises library for
various common tasks related to directional statistics and generative modeling.

Basic Sampling
-------------

Simple examples of sampling from a von Mises distribution:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from jax import random
    from jax_von_mises.sampler import sample_von_mises
    
    # Initialize a random key
    key = random.PRNGKey(42)
    
    # Sample from von Mises with mean direction 0 and concentration 5
    samples = sample_von_mises(key, loc=0.0, concentration=5.0, shape=(10000,))
    
    # Visualize the samples
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='polar')
    ax.hist(samples, bins=50)
    plt.title("von Mises Samples (μ=0, κ=5)")
    plt.show()

Batch Processing with vmap
-------------------------

Using `jax.vmap` to sample efficiently from multiple distributions:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from jax import random, vmap
    from jax_von_mises.sampler import sample_von_mises
    
    # Initialize a random key
    key = random.PRNGKey(42)
    
    # Define a batch of parameters
    batch_size = 5
    means = jnp.array([-jnp.pi/2, 0, jnp.pi/4, jnp.pi/2, jnp.pi])
    concentrations = jnp.array([1.0, 5.0, 10.0, 20.0, 50.0])
    
    # Create a function to sample from each distribution
    def sample_one(key, loc, concentration):
        return sample_von_mises(key, loc, concentration, shape=(1000,))
    
    # Use vmap to apply this function to all parameters at once
    keys = random.split(key, batch_size)
    samples_vmap = vmap(sample_one)(keys, means, concentrations)
    
    # Plot all distributions
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, batch_size, figsize=(15, 3), subplot_kw={'projection': 'polar'})
    for i, ax in enumerate(axes):
        ax.hist(samples_vmap[i], bins=50)
        ax.set_title(f"μ={means[i]:.2f}, κ={concentrations[i]:.1f}")
    plt.tight_layout()
    plt.show()

JIT Compilation for Performance
-----------------------------

Using `jax.jit` to accelerate sampling:

.. code-block:: python

    import time
    import jax
    import jax.numpy as jnp
    from jax import random, jit
    from jax_von_mises.sampler import sample_von_mises
    
    # Create a jitted version of the sampling function
    @jit
    def sample_jitted(key, loc, concentration, shape):
        return sample_von_mises(key, loc, concentration, shape=shape)
    
    # Initialize a random key
    key = random.PRNGKey(42)
    
    # Compare performance
    n_samples = 100000
    
    # Without JIT
    start = time.time()
    _ = sample_von_mises(key, 0.0, 5.0, shape=(n_samples,))
    print(f"Time without JIT: {time.time() - start:.4f} seconds")
    
    # First call with JIT (includes compilation time)
    start = time.time()
    _ = sample_jitted(key, 0.0, 5.0, (n_samples,))
    print(f"Time with JIT (first call): {time.time() - start:.4f} seconds")
    
    # Second call with JIT (compiled version)
    start = time.time()
    _ = sample_jitted(key, 0.0, 5.0, (n_samples,))
    print(f"Time with JIT (second call): {time.time() - start:.4f} seconds")

Neural Network Integration
------------------------

Using von Mises sampling in a neural network:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from jax import random
    import flax.linen as nn
    
    from jax_von_mises.nn.integration import von_mises_layer
    
    # Define a simple directional prediction network
    class DirectionalNetwork(nn.Module):
        @nn.compact
        def __call__(self, x, training=True):
            x = nn.Dense(features=128)(x)
            x = nn.relu(x)
            x = nn.Dense(features=64)(x)
            x = nn.relu(x)
            
            # Output mean and concentration for von Mises
            mean_logits = nn.Dense(features=1)(x)
            concentration = nn.softplus(nn.Dense(features=1)(x))
            
            # Sample using von Mises layer during training
            key = self.make_rng('sample')
            samples, mean = von_mises_layer(
                key, 
                mean_logits, 
                concentration, 
                temperature=1.0,
                training=training
            )
            
            return samples, mean, concentration
    
    # Create model
    model = DirectionalNetwork()
    
    # Initialize parameters with random data
    key1, key2 = random.split(random.PRNGKey(0))
    x = random.normal(key1, (10, 32))  # Batch of 10 examples with 32 features
    params = model.init({'params': key1, 'sample': key2}, x)
    
    # Generate predictions (during training)
    samples, mean, concentration = model.apply(
        params, 
        x, 
        rngs={'sample': random.PRNGKey(1)}, 
        training=True
    )
    
    # Generate predictions (during inference - returns mean direction)
    samples_inf, mean_inf, concentration_inf = model.apply(
        params, 
        x, 
        rngs={'sample': random.PRNGKey(2)}, 
        training=False
    )
    
    print(f"Training mode - samples != mean: {jnp.allclose(samples, mean)}")
    print(f"Inference mode - samples == mean: {jnp.allclose(samples_inf, mean_inf)}")

Circular Statistics
-----------------

Working with circular statistics:

.. code-block:: python

    import jax.numpy as jnp
    from jax_von_mises.utils import (
        circular_mean, 
        circular_variance,
        concentration_to_variance, 
        variance_to_concentration
    )
    
    # Create some sample data (angles in radians)
    angles = jnp.array([0.1, 0.2, -0.1, 0.15, 0.3, 0.0, 0.05])
    
    # Calculate circular statistics
    mean_angle = circular_mean(angles)
    variance = circular_variance(angles)
    
    print(f"Circular mean: {mean_angle:.4f} radians")
    print(f"Circular variance: {variance:.4f}")
    
    # Convert between concentration and variance
    kappa = 5.0
    var = concentration_to_variance(kappa)
    kappa_recovered = variance_to_concentration(var)
    
    print(f"Concentration κ = {kappa:.4f} → Variance = {var:.4f}")
    print(f"Variance = {var:.4f} → Concentration κ = {kappa_recovered:.4f}")

Multi-GPU Sampling with pmap
--------------------------

Utilizing multiple GPUs for parallel sampling:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from jax import random, pmap
    from functools import partial
    
    from jax_von_mises.sampler import sample_von_mises
    
    # Check available devices
    n_devices = jax.device_count()
    print(f"Available devices: {n_devices}")
    
    if n_devices > 1:
        # Define a sampling function for pmap
        @partial(pmap, axis_name='devices')
        def sample_pmap(key, loc, concentration, n_samples):
            return sample_von_mises(key, loc, concentration, shape=(n_samples,))
        
        # Create keys for each device
        keys = random.split(random.PRNGKey(42), n_devices)
        
        # Sample using all devices (each device generates samples_per_device samples)
        samples_per_device = 10000
        samples = sample_pmap(
            keys, 
            jnp.ones(n_devices) * jnp.pi/4,  # Same mean for all devices
            jnp.ones(n_devices) * 10.0,      # Same concentration for all devices
            samples_per_device
        )
        
        # Reshape to combine all samples
        samples = samples.reshape(n_devices * samples_per_device)
        
        print(f"Generated {len(samples)} samples using {n_devices} devices")
    else:
        print("Multi-GPU sampling requires multiple devices") 