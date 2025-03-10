JAX Von Mises Sampling
======================

A JAX-compatible implementation of the von Mises distribution sampling algorithm by Best & Fisher (1979).

This package is specifically designed for use with neural networks and parallel computation on GPUs using ``jax.pmap``.

Features
--------

* Efficient rejection sampling algorithm with ~65.77% acceptance rate even for large concentration values
* Fully compatible with JAX transformations (jit, vmap, pmap)
* Optimized for neural network integration
* Well-tested and numerically stable
* Performance optimized for large-scale sampling operations

Installation
-----------

.. code-block:: bash

    pip install jax-von-mises

For GPU support:

.. code-block:: bash

    # For CUDA 11.x (NVIDIA GPUs)
    pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install jax-von-mises
    
    # For CUDA 12.x (NVIDIA GPUs)
    pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install jax-von-mises
    
    # For ROCm (AMD GPUs)
    pip install "jax[rocm]" -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html
    pip install jax-von-mises

Quick Start
----------

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from jax import random
    from jax_von_mises import sample_von_mises, vmises_log_prob, vmises_entropy
    
    # Generate samples
    key = random.PRNGKey(0)
    loc = 0.0  # mean direction
    concentration = 5.0  # concentration parameter
    samples = sample_von_mises(key, loc, concentration, shape=(1000,))
    
    # JIT compilation (with static_argnums for shape parameter)
    jitted_sampler = jax.jit(sample_von_mises, static_argnums=(3,))
    fast_samples = jitted_sampler(key, loc, concentration, shape=(1000,))
    
    # Calculate log probability and entropy
    log_prob = vmises_log_prob(samples, loc, concentration)
    entropy = vmises_entropy(concentration)  # Entropy depends only on concentration
    
    # Use with automatic differentiation (not possible with SciPy)
    entropy_grad = jax.grad(vmises_entropy)(concentration)
    
    # Neural network integration
    from jax_von_mises.nn import pmap_compatible_von_mises_sampling
    
    def model_fn(params, inputs):
        # Your neural network that outputs (mean, concentration)
        return mean, concentration
    
    # Use with pmap
    parallel_fn = jax.pmap(lambda inputs, key: pmap_compatible_von_mises_sampling(
        model_fn, params, inputs, key
    ))

Documentation Structure
-----------------------

* **Theory Guide**: Mathematical background and algorithm descriptions
* **API Reference**: Complete API documentation with detailed function descriptions
* **Usage Examples**: Practical examples for common use cases
* **JIT Compilation**: Important guide for using JAX's JIT compilation correctly
* **Performance Guide**: Tips for optimizing performance in different scenarios
* **GPU Optimization**: Advanced guide for GPU acceleration
* **Benchmarks**: Performance benchmarking results and profiling tools

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   theory
   guide
   api
   examples/usage_examples
   jit_guide
   performance
   gpu_optimization
   benchmarks

Benchmarking and Profiling
-------------------------

The library includes advanced benchmarking and profiling tools located in the ``benchmarks/`` directory:

- ``profile_gpu.py``: Comprehensive GPU performance profiling
- ``performance_benchmark.py``: Compare performance across different parameters
- ``simple_profile.py``: Quick profiling of core functions

To run the profiling tools:

.. code-block:: bash

    # Run quick profile
    python -m benchmarks.simple_profile
    
    # Run comprehensive GPU profile
    python -m benchmarks.profile_gpu
    
    # Run performance benchmarks
    python -m benchmarks.performance_benchmark

Results are saved to ``profile_results/`` directory with visualizations.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 