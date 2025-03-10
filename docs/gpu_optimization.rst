GPU Optimization Guide
=====================

This guide provides advanced tips for optimizing the JAX von Mises sampling library for GPU execution.

Setting Up GPU Support
---------------------

To leverage GPU acceleration, you need to install the correct JAX variant for your hardware:

.. code-block:: bash

    # For NVIDIA GPUs with CUDA 11.x
    pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    
    # For NVIDIA GPUs with CUDA 12.x
    pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    
    # For AMD GPUs with ROCm
    pip install "jax[rocm]" -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html

For the best setup experience, you can use our included installation script:

.. code-block:: bash

    # Make script executable
    chmod +x install_gpu.sh
    
    # Run the script
    ./install_gpu.sh

Verifying GPU Usage
-----------------

To verify that JAX is using your GPU, run the following Python code:

.. code-block:: python

    import jax
    print(jax.devices())  # Should list your GPU(s)

You can also run our simple profiling script to confirm GPU utilization:

.. code-block:: bash

    python -m benchmarks.simple_profile

If JAX isn't detecting your GPU, troubleshoot with these steps:

1. Verify CUDA/ROCm is installed and in your PATH
2. Check compatible versions (JAX requires specific CUDA/ROCm versions)
3. Set environment variables if needed:

   .. code-block:: bash
   
      # For NVIDIA
      export XLA_FLAGS="--xla_gpu_cuda_data_dir=/path/to/cuda"
      
      # For AMD
      export XLA_FLAGS="--xla_gpu_platform_id=ROCM"

XLA Optimization Flags
---------------------

JAX uses XLA (Accelerated Linear Algebra) as its compiler backend. You can optimize XLA performance with these flags:

.. code-block:: bash

    # Enable autotune (finds optimal algorithms)
    export XLA_FLAGS="--xla_gpu_autotune=true"
    
    # Increase GPU memory limit (adjust based on your GPU)
    export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=8"
    
    # For multi-GPU setups
    export XLA_FLAGS="--xla_gpu_enable_fast_min_max=true"

Multiple flags can be combined:

.. code-block:: bash

    export XLA_FLAGS="--xla_gpu_autotune=true --xla_gpu_cuda_data_dir=/path/to/cuda"

Memory Management
---------------

GPU memory can be a bottleneck for large sampling operations. Optimize memory usage by:

1. **Using fp32 precision** (default in JAX)
2. **Batching operations** for very large sample sizes
3. **Explicitly clearing results** when no longer needed
4. **Using memory-mapped arrays** for huge datasets

Example of batched processing:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from jax import random
    from jax_von_mises import sample_von_mises
    
    # Generate 10 million samples in batches of 1 million
    key = random.PRNGKey(0)
    total_samples = 10_000_000
    batch_size = 1_000_000
    num_batches = total_samples // batch_size
    
    all_samples = []
    for i in range(num_batches):
        subkey = random.fold_in(key, i)
        batch = sample_von_mises(subkey, 0.0, 5.0, shape=(batch_size,))
        # Process batch
        all_samples.append(batch)
        # Optional: Move to CPU to free GPU memory
        # all_samples.append(jax.device_get(batch))
    
    # Combine results if needed
    combined = jnp.concatenate(all_samples)

Multi-GPU Strategies
------------------

For systems with multiple GPUs, use these strategies for maximum performance:

1. **Data Parallelism with pmap**

   .. code-block:: python
   
       from functools import partial
       from jax import pmap
       
       # Define a pmap-compatible sampling function
       @partial(pmap, axis_name='devices')
       def sample_pmap(key, loc, concentration, n_samples):
           return sample_von_mises(key, loc, concentration, shape=(n_samples,))
           
       # Split across devices
       n_devices = jax.device_count()
       keys = random.split(key, n_devices)
       locs = jnp.ones(n_devices) * 0.0
       concentrations = jnp.ones(n_devices) * 5.0
       
       # Run in parallel (100k samples per device)
       samples = sample_pmap(keys, locs, concentrations, 100000)

2. **Pipeline Parallelism for Complex Workflows**

   For complex pipelines where you generate samples and then process them:
   
   .. code-block:: python
   
       # On device 0: Generate samples
       with jax.devices()[0]:
           samples = sample_von_mises(key0, loc, concentration, shape=(100000,))
           
       # On device 1: Process samples
       with jax.devices()[1]:
           processed = process_fn(samples)

3. **Asynchronous Operations**

   JAX operations are asynchronous by default. Use `block_until_ready()` to explicitly synchronize:
   
   .. code-block:: python
   
       # Start a long computation
       result = large_sampling_operation(key, loc, concentration)
       
       # Do other work here while sampling happens in background
       
       # Wait for completion when needed
       result.block_until_ready()

Compiled Function Caching
-----------------------

JAX recompiles functions when their signatures change. Use the compilation cache to speed up repeated runs:

.. code-block:: python

    from jax.experimental.compilation_cache import compilation_cache as cc
    
    # Initialize cache (do this once at program start)
    cc.initialize_cache("./jax_cache")
    
    # Now compiled functions will be cached between runs

Profiling and Benchmarking
------------------------

Use our profiling tools to identify bottlenecks:

.. code-block:: bash

    # Full GPU profiling with visualizations
    python -m benchmarks.profile_gpu
    
    # Performance benchmarks across parameters
    python -m benchmarks.performance_benchmark

You can also use JAX's built-in profiling:

.. code-block:: python

    from jax.profiler import start_trace, stop_trace
    
    # Start profiling
    start_trace("./profile_results")
    
    # Run your code
    samples = sample_von_mises(key, loc, concentration, shape=(100000,))
    samples.block_until_ready()
    
    # Stop profiling
    stop_trace()

View the results with TensorBoard:

.. code-block:: bash

    tensorboard --logdir=./profile_results

Common GPU-Specific Issues
------------------------

1. **Out of Memory Errors**

   Solution: Reduce batch size, use memory batching, or try releasing memory:
   
   .. code-block:: python
   
       # Clear JAX's internal cache
       from jax.lib import xla_bridge
       xla_bridge.get_backend().clear_compile_cache()
       
       # Force garbage collection
       import gc
       gc.collect()

2. **Slow First Run**

   This is normal due to compilation. Solutions:
   
   - Use the compilation cache
   - Add a warmup run with small inputs
   - JIT compile functions in advance

3. **Different Results on CPU vs. GPU**

   May be due to different floating-point behavior:
   
   - Use `jnp.isclose()` with appropriate tolerance
   - Consider setting a fixed random seed

4. **Multi-GPU Synchronization Issues**

   For complex multi-GPU workflows:
   
   - Use explicit `block_until_ready()`
   - Consider SPMD (Single Program Multiple Data) style with pmap
   
5. **GPU Utilization Too Low**

   - Increase batch size
   - Ensure operations are large enough to benefit from GPU
   - Use the XLA autotune flag

Advanced Optimization Techniques
-----------------------------

For squeezing maximum performance:

1. **Kernel Fusion**

   JAX automatically fuses operations, but you can help by:
   
   - Grouping related operations in JIT-compiled functions
   - Avoiding unnecessary intermediate results
   
2. **Custom XLA Operations**

   For experts, you can write custom XLA operations:
   
   .. code-block:: python
   
       from jax._src.lib import xla_client
       xla_client.register_custom_call_target(...)  # Advanced usage

3. **Mixed Precision**

   While JAX defaults to fp32, you can use reduced precision:
   
   .. code-block:: python
   
       from jax.experimental import enable_x64
       
       # Disable double precision (use fp32)
       enable_x64(False)  # This is the default
       
       # For some hardware, you can use fp16
       x = x.astype(jnp.float16)

4. **Optimizing Small Batch Performance**

   For small batches, ensure they're large enough for GPU efficiency:
   
   .. code-block:: python
   
       # Less efficient for small batches
       for i in range(100):
           sample_von_mises(key, loc, concentration, shape=(10,))
       
       # More efficient
       sample_von_mises(key, loc, concentration, shape=(1000,))

5. **Device Memory vs. Host Memory Transfers**

   Minimize CPU-GPU transfers:
   
   .. code-block:: python
   
       # Inefficient (multiple transfers)
       for i in range(10):
           x = jax.device_get(compute_on_gpu())  # Transfers to CPU
           y = jax.device_put(process_on_cpu(x))  # Transfers to GPU
       
       # More efficient (keep on GPU)
       @jax.jit
       def full_pipeline(x):
           for i in range(10):
               x = process_step(x)
           return x
       
       result = full_pipeline(inputs)
       # Only transfer final result
       result_cpu = jax.device_get(result) 