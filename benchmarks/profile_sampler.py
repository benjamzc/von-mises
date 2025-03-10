"""
Runtime profiling for von Mises sampling functions.

This script uses JAX's built-in profiling capabilities to identify performance
bottlenecks in the sampling implementation.
"""

import os
import time
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import random, jit, vmap, pmap, profiler

# Optional TensorFlow import
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Some profiling features will be limited.")

from jax_von_mises.sampler import sample_von_mises, vmises_log_prob, compute_p


def profile_main_functions(save_dir='profile_results'):
    """Profile the main sampling functions individually."""
    if not TF_AVAILABLE:
        print("Skipping TensorFlow-based profiling as TensorFlow is not available.")
        return
        
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up profiling
    key = random.PRNGKey(42)
    
    # Profile vmises_log_prob
    print("Profiling vmises_log_prob...")
    x = jnp.linspace(-jnp.pi, jnp.pi, 10000)
    loc = 0.0
    concentration = 5.0
    
    # JIT compile for fair comparison
    vmises_log_prob_jit = jit(vmises_log_prob)
    
    # Warm up
    _ = vmises_log_prob_jit(x, loc, concentration)
    
    # Profile with trace
    with tf.profiler.experimental.Profile(save_dir + '/vmises_log_prob_trace'):
        for _ in range(100):
            _ = vmises_log_prob_jit(x, loc, concentration).block_until_ready()
    
    # Profile compute_p
    print("Profiling compute_p...")
    kappa_values = jnp.logspace(-3, 3, 10000)
    
    # JIT compile
    compute_p_jit = jit(compute_p)
    
    # Warm up
    _ = compute_p_jit(kappa_values)
    
    # Profile with trace
    with tf.profiler.experimental.Profile(save_dir + '/compute_p_trace'):
        for _ in range(100):
            _ = compute_p_jit(kappa_values).block_until_ready()
    
    # Profile sample_von_mises with different concentration values
    print("Profiling sample_von_mises...")
    
    # JIT compile with static_argnums for shape
    sample_jit = jit(sample_von_mises, static_argnums=(3,))
    
    # Define shape tuple
    shape = (10000,)
    
    # Warm up
    _ = sample_jit(key, 0.0, 5.0, shape)
    
    # Profile with different concentration values
    for kappa in [0.1, 1.0, 10.0, 100.0]:
        trace_name = f'/sample_von_mises_kappa_{kappa}_trace'
        with tf.profiler.experimental.Profile(save_dir + trace_name):
            for _ in range(10):  # Fewer iterations due to higher computation cost
                _ = sample_jit(key, 0.0, kappa, shape).block_until_ready()
    
    print(f"Basic profiling complete. Results saved to {save_dir}")
    print(f"View the traces using TensorBoard:")
    print(f"  tensorboard --logdir={save_dir}")


def profile_timing_measurements():
    """Simple timing measurements for core functions without TensorFlow dependency."""
    print("\n=== Basic Timing Measurements ===")
    key = random.PRNGKey(42)
    
    # Measure vmises_log_prob
    x = jnp.linspace(-jnp.pi, jnp.pi, 10000)
    loc = 0.0
    concentration = 5.0
    
    # JIT compile
    vmises_log_prob_jit = jit(vmises_log_prob)
    
    # Warm up
    _ = vmises_log_prob_jit(x, loc, concentration).block_until_ready()
    
    # Time execution
    start = time.time()
    for _ in range(100):
        _ = vmises_log_prob_jit(x, loc, concentration).block_until_ready()
    elapsed = time.time() - start
    print(f"vmises_log_prob: {elapsed:.4f}s for 100 iterations ({elapsed/100*1000:.2f}ms per call)")
    
    # Measure compute_p
    kappa_values = jnp.logspace(-3, 3, 10000)
    
    # JIT compile
    compute_p_jit = jit(compute_p)
    
    # Warm up
    _ = compute_p_jit(kappa_values).block_until_ready()
    
    # Time execution
    start = time.time()
    for _ in range(100):
        _ = compute_p_jit(kappa_values).block_until_ready()
    elapsed = time.time() - start
    print(f"compute_p: {elapsed:.4f}s for 100 iterations ({elapsed/100*1000:.2f}ms per call)")
    
    # Measure sample_von_mises with different concentration values
    sample_jit = jit(sample_von_mises, static_argnums=(3,))
    
    # Define shape tuple
    shape = (10000,)
    
    # Warm up
    _ = sample_jit(key, 0.0, 5.0, shape).block_until_ready()
    
    # Time execution for different kappa values
    for kappa in [0.1, 1.0, 10.0, 100.0]:
        start = time.time()
        for _ in range(10):
            _ = sample_jit(key, 0.0, kappa, shape).block_until_ready()
        elapsed = time.time() - start
        print(f"sample_von_mises (κ={kappa}): {elapsed:.4f}s for 10 iterations ({elapsed/10*1000:.2f}ms per call)")


def profile_batch_processing():
    """Profile batch processing performance."""
    print("\n=== Batch Processing Performance ===")
    key = random.PRNGKey(42)
    
    # Create large batch of parameters
    batch_size = 1000
    locs = jnp.linspace(-jnp.pi, jnp.pi, batch_size)
    concentrations = jnp.ones(batch_size) * 5.0
    
    # Define function for a single sample with fixed shape
    def sample_one(key, loc, concentration):
        shape = (100,)
        return sample_von_mises(key, loc, concentration, shape=shape)
    
    # Create vmap version
    sample_vmap = vmap(sample_one)
    
    # JIT compile
    sample_vmap_jit = jit(sample_vmap)
    
    # Generate keys
    keys = random.split(key, batch_size)
    
    # Warm up
    _ = sample_vmap_jit(keys, locs, concentrations)
    
    with jax.disable_jit():
        print("Profiling without JIT...")
        start_time = time.time()
        # This will be slow but shows Python overhead
        _ = sample_vmap(keys, locs, concentrations)
        print(f"Time without JIT: {time.time() - start_time:.2f} seconds")
    
    print("Profiling with JIT...")
    start_time = time.time()
    _ = sample_vmap_jit(keys, locs, concentrations).block_until_ready()
    print(f"Time with JIT: {time.time() - start_time:.2f} seconds")
    
    # Calculate speedup
    speedup = (time.time() - start_time) / (time.time() - start_time)
    print(f"JIT speedup: {speedup:.1f}x")
    
    # Test scaling with batch size
    print("\nScaling with batch size:")
    batch_sizes = [10, 100, 1000]
    for size in batch_sizes:
        # Create batch
        sub_locs = locs[:size]
        sub_concentrations = concentrations[:size]
        sub_keys = keys[:size]
        
        # Time execution
        start_time = time.time()
        _ = sample_vmap_jit(sub_keys, sub_locs, sub_concentrations).block_until_ready()
        elapsed = time.time() - start_time
        
        print(f"Batch size {size}: {elapsed:.4f} seconds ({size/elapsed:.0f} samples/sec)")


def profile_memory_leaks(n_iterations=50):
    """Check for potential memory leaks during repeated sampling."""
    print("\n=== Memory Leak Check ===")
    try:
        import psutil
    except ImportError:
        print("psutil not available. Install with 'pip install psutil'")
        return
    
    process = psutil.Process(os.getpid())
    key = random.PRNGKey(42)
    
    # JIT compile function once with static_argnums
    sample_jit = jit(sample_von_mises, static_argnums=(3,))
    
    # Define shape tuple
    shape = (10000,)
    
    # Warm up
    _ = sample_jit(key, 0.0, 5.0, shape)
    
    # Track memory usage over iterations
    memory_usage = []
    for i in range(n_iterations):
        # Split key for each iteration
        key, subkey = random.split(key)
        
        # Run sampling
        _ = sample_jit(subkey, 0.0, 5.0, shape).block_until_ready()
        
        # Record memory
        memory_mb = process.memory_info().rss / 1024 / 1024
        memory_usage.append(memory_mb)
        
        if i % 10 == 0:
            print(f"Iteration {i}: Memory usage {memory_mb:.2f} MB")
    
    # Plot memory usage
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_iterations), memory_usage)
    plt.xlabel('Iteration')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Over Repeated Sampling Operations')
    plt.grid(True)
    plt.savefig('memory_usage_profile.png', dpi=300, bbox_inches='tight')
    
    # Check for trend
    if len(memory_usage) > 10:
        early_avg = np.mean(memory_usage[:10])
        late_avg = np.mean(memory_usage[-10:])
        change = late_avg - early_avg
        
        print(f"Memory change: {change:.2f} MB ({change/early_avg*100:.2f}%)")
        if change / early_avg > 0.05:  # More than 5% growth
            print("WARNING: Possible memory leak detected")
        else:
            print("No significant memory growth detected")


def profile_gpu_utilization():
    """Profile GPU utilization during sampling operations."""
    print("\n=== GPU Performance Scaling ===")
    # Check if GPU is available
    if len(jax.devices('gpu')) == 0:
        print("No GPU devices found. Skipping GPU profiling.")
        return
    
    print("Profiling GPU operations...")
    
    # Set up parameters
    key = random.PRNGKey(42)
    
    # Create a range of sample sizes
    sample_sizes = [1000, 10000, 100000, 1000000]
    
    # JIT the sampling function with static_argnums
    sample_jit = jit(sample_von_mises, static_argnums=(3,))
    
    # Warm up
    _ = sample_jit(key, 0.0, 5.0, (1000,))
    
    # Profile each sample size
    results = []
    for size in sample_sizes:
        print(f"Testing sample size: {size}")
        
        # Instead of save_device_memory_profile, use psutil to measure memory before and after
        try:
            import psutil
            process = psutil.Process(os.getpid())
            before_mem = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            print("psutil not available. Install with 'pip install psutil'")
            before_mem = 0
        
        # Create shape tuple
        shape = (size,)
        
        # Run sampling with timing
        start_time = time.time()
        result = sample_jit(key, 0.0, 5.0, shape)
        result.block_until_ready()  # Ensure the computation completes
        execution_time = time.time() - start_time
        
        # Measure memory after
        try:
            after_mem = process.memory_info().rss / (1024 * 1024)
            diff_mb = after_mem - before_mem
            print(f"  Memory change: {diff_mb:.2f} MB")
        except:
            diff_mb = 0
            pass
        
        throughput = size/execution_time
        print(f"  Execution time: {execution_time:.4f} seconds")
        print(f"  Throughput: {throughput:.0f} samples/second")
        
        results.append({
            'size': size,
            'time': execution_time,
            'throughput': throughput,
            'memory_delta_mb': diff_mb
        })
    
    # Plot results
    plt.figure(figsize=(10, 6))
    sizes = [r['size'] for r in results]
    throughputs = [r['throughput'] for r in results]
    plt.plot(sizes, throughputs, 'o-')
    plt.xscale('log')
    plt.xlabel('Sample Size')
    plt.ylabel('Throughput (samples/second)')
    plt.title('Sampling Performance Scaling')
    plt.grid(True)
    plt.savefig('gpu_scaling.png', dpi=300, bbox_inches='tight')
    print("Performance scaling plot saved to 'gpu_scaling.png'")


def run_all_profiles():
    """Run all profiling functions."""
    print("=== Starting JAX Performance Profiling ===")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    
    # Run profiling functions that don't depend on TensorFlow
    profile_timing_measurements()
    profile_batch_processing()
    profile_memory_leaks(n_iterations=50)
    profile_gpu_utilization()
    
    # Run TensorFlow profiler only if TensorFlow is available
    if TF_AVAILABLE:
        try:
            profile_main_functions()
        except Exception as e:
            print(f"TensorFlow profiler error: {e}")
            print("Skipping TensorFlow-based profiling. Basic profiling still completed.")
    
    print("=== Profiling Complete ===")
    print("Memory usage and performance plots saved to current directory.")
    if TF_AVAILABLE:
        print("If TensorFlow profiling was successful, view the traces using:")
        print("  tensorboard --logdir=profile_results")


if __name__ == "__main__":
    run_all_profiles() 