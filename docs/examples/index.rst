Examples
========

This section contains examples showing how to use the jax-von-mises package for various tasks.

.. toctree::
   :maxdepth: 1
   
   neural_network_integration
   parallel_sampling

Basic Usage
----------

Here's a simple example of generating samples from a von Mises distribution:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from jax import random
    import matplotlib.pyplot as plt
    from jax_von_mises import sample_von_mises
    
    # Set up parameters
    key = random.PRNGKey(42)
    loc = 0.0  # mean direction
    concentration = 5.0  # concentration parameter (higher = more concentrated)
    
    # Generate samples
    samples = sample_von_mises(key, loc, concentration, shape=(10000,))
    
    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.hist(samples, bins=50, density=True)
    plt.title(f'Von Mises Distribution (μ={loc}, κ={concentration})')
    plt.xlabel('Angle (radians)')
    plt.ylabel('Density')
    plt.xlim(-jnp.pi, jnp.pi)
    plt.show()

Working with Batched Parameters
------------------------------

The implementation supports batched parameters, which is useful when working with outputs from neural networks:

.. code-block:: python

    # Batched parameters
    locs = jnp.array([0.0, jnp.pi/2, jnp.pi])
    concentrations = jnp.array([1.0, 5.0, 10.0])
    
    # Generate batched samples
    batched_samples = sample_von_mises(key, locs, concentrations, shape=(1000, 3))
    
    # Plot separate histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (loc, conc) in enumerate(zip(locs, concentrations)):
        axes[i].hist(batched_samples[:, i], bins=30, density=True)
        axes[i].set_title(f'Von Mises (μ={loc:.2f}, κ={conc:.1f})')
        axes[i].set_xlabel('Angle (radians)')
        axes[i].set_xlim(-jnp.pi, jnp.pi)
    plt.tight_layout()
    plt.show()

Using JAX Transformations
------------------------

The implementation is compatible with JAX transformations:

.. code-block:: python

    # JIT compilation
    jitted_sampler = jax.jit(sample_von_mises)
    
    # Time comparison
    %timeit sample_von_mises(key, 0.0, 2.0, shape=(10000,))
    %timeit jitted_sampler(key, 0.0, 2.0, shape=(10000,))
    
    # Vectorization with vmap
    vmapped_sampler = jax.vmap(
        lambda k, l, c: sample_von_mises(k, l, c, shape=(1000,)),
        in_axes=(0, 0, 0)
    )
    
    # Generate keys and parameters for 5 distributions
    keys = random.split(key, 5)
    locs = jnp.linspace(-jnp.pi/2, jnp.pi/2, 5)
    concs = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
    
    # Sample in parallel
    parallel_samples = vmapped_sampler(keys, locs, concs)
    print(parallel_samples.shape)  # (5, 1000) 