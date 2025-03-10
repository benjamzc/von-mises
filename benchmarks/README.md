# JAX Von Mises Sampling Benchmarks

This directory contains tools for benchmarking and profiling the performance of the JAX von Mises sampling implementation.

## Installation

To install the benchmarking dependencies, use:

```bash
pip install jax-von-mises[benchmark]
```

## Available Tools

### Performance Benchmarking

The `performance_benchmark.py` script measures and visualizes the performance of the sampling functions across various scenarios:

- JIT vs No JIT comparison
- Concentration parameter impact
- Batch processing with vmap
- Multi-device execution with pmap
- Neural network integration performance
- Memory usage profiling

To run all benchmarks:

```bash
python -m benchmarks.performance_benchmark
```

Results will be saved to the `benchmark_results` directory as CSV files and visualization plots.

### Runtime Profiling

The `profile_sampler.py` script uses JAX's built-in profiling capabilities to identify performance bottlenecks:

- Individual function profiling
- Batch processing profiling
- Memory leak detection
- GPU utilization profiling

To run the profiler:

```bash
python -m benchmarks.profile_sampler
```

The profiling results will be saved to the `profile_results` directory and can be viewed using TensorBoard:

```bash
tensorboard --logdir=profile_results
```

## Custom Benchmarking

You can also import and use the benchmarking functions directly in your own code:

```python
from benchmarks.performance_benchmark import benchmark_jit_vs_nojit, benchmark_vmap_scaling

# Run specific benchmarks
jit_results = benchmark_jit_vs_nojit([1000, 10000, 100000])
vmap_results = benchmark_vmap_scaling([10, 100, 1000])

# Analyze results
print(f"JIT speedup: {jit_results['speedup'].mean():.2f}x")
print(f"vmap speedup: {vmap_results['speedup'].mean():.2f}x")
```

## GPU-Specific Optimizations

For CUDA-specific optimizations, you can set XLA flags:

```bash
XLA_FLAGS="--xla_gpu_cuda_data_dir=/path/to/cuda --xla_gpu_autotune=true" python -m benchmarks.performance_benchmark
```

## Memory Profiling

For detailed memory profiling with larger sample sizes, modify the script parameters:

```python
# In profile_sampler.py
sample_sizes = [10, 100, 1000, 10000, 100000, 1000000, 10000000]  # Increased sizes
```

Note that very large sample sizes may require significant memory. 