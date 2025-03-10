Performance Benchmarks
=====================

This section provides benchmark results and guidance on performance optimization for the JAX von Mises sampling library.

Benchmark Methodology
-------------------

All benchmarks were run using the following tools:

1. ``profile_gpu.py``: Comprehensive profiling across multiple parameters
2. ``performance_benchmark.py``: Focused benchmarks for specific performance aspects
3. ``simple_profile.py``: Quick profiling of core functions

System Configurations
^^^^^^^^^^^^^^^^^^^

Benchmarks were run on the following configurations:

- **NVIDIA GPU**: Single NVIDIA RTX 3090 with CUDA 11.8
- **AMD GPU**: Single AMD MI210 with ROCm 5.4
- **CPU**: Intel Xeon 8380 with 40 cores

Each benchmark was run with 5 repetitions, and results were averaged to ensure consistency.

Core Function Performance
-----------------------

The von Mises sampling implementation consists of several core functions, each with different performance characteristics:

.. figure:: _static/function_timings.png
   :width: 600
   :align: center
   :alt: Function Timing Comparison
   
   Execution time comparison for core functions (lower is better)

Function performance breakdown:

1. ``vmises_log_prob``: Very efficient, typically <0.01ms per call
2. ``compute_p``: Extremely fast, typically <0.005ms per call
3. ``sample_von_mises``: Performance varies based on parameters and configuration

JIT Compilation Impact
--------------------

JIT compilation provides significant speedups for all functions:

.. figure:: _static/jit_vs_nojit.png
   :width: 600
   :align: center
   :alt: JIT vs Non-JIT Performance
   
   Impact of JIT compilation on sampling performance (higher is better)

Key observations:

- JIT compilation provides 100-10,000x speedup depending on the function
- First call to a JIT-compiled function incurs compilation overhead
- Subsequent calls are dramatically faster
- GPU execution benefits even more from JIT compilation

Concentration Parameter Impact
----------------------------

The concentration parameter (κ) significantly affects sampling performance:

.. figure:: _static/concentration_impact.png
   :width: 600
   :align: center
   :alt: Concentration Parameter Impact
   
   Impact of concentration parameter on sampling performance (higher is better)

Key findings:

- Low concentration values (κ < 1.0) achieve highest performance
- Performance gradually decreases as concentration increases
- Very high concentration values (κ > 100) see improved performance due to normal approximation
- The implementation optimizes for both extreme cases (very low and very high concentration)

Batch Processing Performance
--------------------------

Vectorized operations using ``vmap`` show significant performance improvements for batch processing:

.. figure:: _static/batch_scaling.png
   :width: 600
   :align: center
   :alt: Batch Size Scaling
   
   Performance scaling with batch size using vmap (higher is better)

Key insights:

- Vectorization provides near-linear speedup for small to medium batch sizes
- Optimal batch size varies by hardware but is typically 64-512
- Very large batch sizes may see diminishing returns due to memory bandwidth limitations

Sample Size Scaling
-----------------

The performance scaling with sample size shows the efficiency of the implementation:

.. figure:: _static/sample_size_scaling.png
   :width: 600
   :align: center
   :alt: Sample Size Scaling
   
   Performance scaling with sample size (higher is better)

Analysis:

- Performance increases with sample size up to around 10,000 samples
- For very large sample sizes, performance stabilizes
- GPU execution shows better scaling for large sample sizes compared to CPU
- Memory limitations become significant for extremely large sample sizes (>10M)

Multi-Device Scaling
------------------

For systems with multiple GPUs, parallel execution using ``pmap`` provides additional speedup:

.. figure:: _static/multi_device_scaling.png
   :width: 800
   :align: center
   :alt: Multi-Device Scaling
   
   Performance scaling with multiple devices (higher is better)

Findings:

- Near-linear scaling with the number of devices for moderate sample sizes
- Diminishing returns due to communication overhead for very small sample sizes
- Efficiency per device decreases slightly as more devices are added
- Optimal workload size increases with device count

Optimization Recommendations
--------------------------

Based on the benchmark results, we recommend the following optimization strategies:

1. **Always use JIT**: Ensure all sampling functions are JIT-compiled for maximum performance
2. **Batch processing**: Use ``vmap`` for processing multiple distributions simultaneously
3. **Multi-device execution**: For large workloads, distribute across multiple GPUs with ``pmap``
4. **Optimal concentration handling**: Consider separating processing for very high and very low concentration values
5. **Memory management**: For very large sample sizes, generate in batches to avoid memory issues
6. **GPU optimization**: Set XLA flags for your specific GPU architecture:

   .. code-block:: bash

      XLA_FLAGS="--xla_gpu_autotune=true --xla_gpu_cuda_data_dir=/path/to/cuda" python your_script.py

Running Your Own Benchmarks
-------------------------

To run benchmarks on your system:

1. Install the package with benchmark dependencies:

   .. code-block:: bash

      pip install "jax-von-mises[benchmark]"

2. Run the benchmarking scripts:

   .. code-block:: bash

      python -m benchmarks.profile_gpu
      python -m benchmarks.performance_benchmark

3. View results in the ``profile_results/`` directory.

For custom benchmarks, you can import and use the benchmarking functions directly:

.. code-block:: python

   from benchmarks.performance_benchmark import benchmark_jit_vs_nojit
   
   results = benchmark_jit_vs_nojit([100, 1000, 10000], n_runs=5)
   print(results) 