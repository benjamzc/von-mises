Performance Guide
================

This guide provides practical advice for optimizing the performance of JAX von Mises sampling in your applications.

JIT Compilation
--------------

Always use JAX's just-in-time compilation (``jax.jit``) for repeated sampling operations:

.. code-block:: python

    from jax import jit
    from jax_von_mises import sample_von_mises
    
    # Create a jitted sampling function
    sample_jit = jit(sample_von_mises)
    
    # First call includes compilation overhead
    samples = sample_jit(key, loc, concentration, shape=(10000,))
    
    # Subsequent calls are much faster
    more_samples = sample_jit(new_key, loc, concentration, shape=(10000,))

The first call to a jitted function includes compilation time, but subsequent calls will be significantly faster.

Batch Processing
---------------

For generating multiple sets of samples with different parameters, use ``jax.vmap`` rather than loops:

.. code-block:: python

    from jax import vmap
    
    # Define a function to generate a single batch of samples
    def sample_fn(key, loc, concentration):
        return sample_von_mises(key, loc, concentration, shape=(1000,))
    
    # Create a vectorized version that handles batches
    sample_vmap = vmap(sample_fn)
    
    # Use with batches of parameters
    batch_size = 100
    keys = random.split(key, batch_size)
    locs = jnp.linspace(-jnp.pi, jnp.pi, batch_size)
    concentrations = jnp.ones(batch_size) * 5.0
    
    # Generate all samples at once (much faster than a loop)
    all_samples = sample_vmap(keys, locs, concentrations)

``vmap`` can provide near-linear scaling with batch size and is much more efficient than Python loops.

Multi-Device Execution
---------------------

For large-scale sampling on multi-GPU systems, use ``jax.pmap``:

.. code-block:: python

    from jax import pmap
    from functools import partial
    
    # Define a pmap-compatible sampling function
    @partial(pmap, axis_name='devices')
    def sample_pmap(key, loc, concentration, n_samples):
        return sample_von_mises(key, loc, concentration, shape=(n_samples,))
    
    # Create one key per device
    n_devices = jax.device_count()
    keys = random.split(random.PRNGKey(0), n_devices)
    
    # Run on all devices in parallel
    samples_per_device = 10000
    samples = sample_pmap(
        keys, 
        jnp.ones(n_devices) * jnp.pi/4,  # Same loc for all devices
        jnp.ones(n_devices) * 5.0,       # Same concentration for all devices
        samples_per_device
    )

``pmap`` will distribute the computation across all available devices, providing significant speedup for large sample sizes.

Memory Considerations
-------------------

When generating very large numbers of samples, be mindful of memory usage:

1. **Generate in Batches**: For extremely large sample sizes (e.g., millions), consider generating samples in smaller batches to avoid memory issues.

2. **Use 32-bit Precision**: By default, JAX uses 32-bit floating point. This is usually sufficient for most applications and reduces memory usage compared to 64-bit.

3. **Release Memory**: When working with multiple large sample batches, explicitly delete arrays when no longer needed:

   .. code-block:: python
   
       # Generate first batch
       samples1 = sample_jit(key1, loc, concentration, shape=(1000000,))
       # Process samples1
       process_samples(samples1)
       # Delete to free memory
       del samples1
       
       # Generate next batch
       samples2 = sample_jit(key2, loc, concentration, shape=(1000000,))

Concentration Parameter Considerations
------------------------------------

The performance of the sampling algorithm varies with the concentration parameter (κ):

- **Very Small κ (< 0.1)**: Sampling is fast as it approaches uniform sampling
- **Medium κ (0.1-50)**: Uses the Best-Fisher rejection sampling algorithm
- **Large κ (> 100)**: Uses a normal approximation, which is faster for extreme concentrations

When possible, batch together similar concentration values for optimal performance.

Neural Network Integration
------------------------

When using the von Mises sampling layer in neural networks:

1. **Batch Dimensions**: Ensure your batch dimensions are compatible across inputs.

2. **Temperature Scaling**: Use the temperature parameter to control exploration during training:

   .. code-block:: python
   
       # Higher temperature (> 1.0) increases diversity
       samples, mean = von_mises_layer(key, mean_logits, concentration, temperature=2.0)
       
       # Lower temperature (< 1.0) reduces diversity
       samples, mean = von_mises_layer(key, mean_logits, concentration, temperature=0.5)

3. **Inference Mode**: During inference, set ``training=False`` to bypass sampling and directly use the mean direction:

   .. code-block:: python
   
       # Training mode - samples from distribution
       train_samples, _ = von_mises_layer(key, mean_logits, concentration, training=True)
       
       # Inference mode - uses mean direction directly
       inference_samples, _ = von_mises_layer(key, mean_logits, concentration, training=False)

Profiling and Benchmarking
-------------------------

The library includes tools for profiling and benchmarking in the ``benchmarks`` directory:

1. **Performance Benchmark**: Use ``performance_benchmark.py`` to measure performance across different parameters and batch sizes.

2. **Profiling**: Use ``profile_sampler.py`` to identify bottlenecks in your specific application.

Example benchmarking command:

.. code-block:: bash

    python -m benchmarks.performance_benchmark

    # Or with specific JAX flags
    XLA_FLAGS="--xla_gpu_cuda_data_dir=/path/to/cuda" python -m benchmarks.performance_benchmark 