"""
Benchmarking script for JAX von Mises sampling performance.

This script profiles the performance of the von Mises sampling functions
under various conditions and configurations.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

import jax
import jax.numpy as jnp
from jax import random, jit, vmap, pmap

from jax_von_mises.sampler import sample_von_mises, vmises_log_prob
from jax_von_mises.nn.integration import von_mises_layer


def benchmark_jit_vs_nojit(n_samples_list, n_runs=5):
    """Benchmark JIT compilation vs no JIT."""
    results = []
    key = random.PRNGKey(42)
    
    # Define the jitted function with static_argnums for shape parameter
    sample_jitted = jit(sample_von_mises, static_argnums=(3,))
    
    for n_samples in n_samples_list:
        # Create shape tuple
        shape = (n_samples,)
        
        # Benchmark without JIT
        nojit_times = []
        for _ in range(n_runs):
            start = time.time()
            _ = sample_von_mises(key, 0.0, 5.0, shape=shape)
            nojit_times.append(time.time() - start)
        
        # Benchmark with JIT
        # First call (includes compilation)
        start = time.time()
        _ = sample_jitted(key, 0.0, 5.0, shape)
        jit_first_time = time.time() - start
        
        # Subsequent calls
        jit_times = []
        for _ in range(n_runs):
            start = time.time()
            _ = sample_jitted(key, 0.0, 5.0, shape)
            jit_times.append(time.time() - start)
        
        results.append({
            'n_samples': n_samples,
            'nojit_mean': np.mean(nojit_times),
            'nojit_std': np.std(nojit_times),
            'jit_first': jit_first_time,
            'jit_mean': np.mean(jit_times),
            'jit_std': np.std(jit_times),
            'speedup': np.mean(nojit_times) / np.mean(jit_times)
        })
        
        print(f"Sample size {n_samples}: Speedup with JIT: {results[-1]['speedup']:.2f}x")
    
    return pd.DataFrame(results)


def benchmark_concentration_impact(kappa_list, n_samples=10000, n_runs=5):
    """Benchmark impact of concentration parameter on performance."""
    results = []
    key = random.PRNGKey(42)
    
    # Use JIT for consistent measurement with static_argnums
    sample_jitted = jit(sample_von_mises, static_argnums=(3,))
    
    # Create shape tuple
    shape = (n_samples,)
    
    # Warmup
    _ = sample_jitted(key, 0.0, 5.0, shape)
    
    for kappa in kappa_list:
        times = []
        for _ in range(n_runs):
            start = time.time()
            _ = sample_jitted(key, 0.0, kappa, shape)
            times.append(time.time() - start)
        
        results.append({
            'concentration': kappa,
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'samples_per_sec': n_samples / np.mean(times)
        })
        
        print(f"Concentration κ={kappa}: {results[-1]['samples_per_sec']:.0f} samples/second")
    
    return pd.DataFrame(results)


def benchmark_vmap_scaling(batch_size_list, samples_per_item=100, n_runs=3):
    """Benchmark vmap performance scaling with batch size."""
    results = []
    key = random.PRNGKey(42)
    
    # Define the sampling function for a single item with fixed shape
    def sample_fn(key, loc, concentration):
        return sample_von_mises(key, loc, concentration, shape=(samples_per_item,))
    
    # Jitted vmap version
    sample_vmap_jit = jit(vmap(sample_fn))
    
    # Warmup
    keys = random.split(key, 10)
    locs = jnp.zeros(10)
    concentrations = jnp.ones(10) * 5.0
    _ = sample_vmap_jit(keys, locs, concentrations)
    
    for batch_size in batch_size_list:
        # Create batch of keys, locs, and concentrations
        keys = random.split(key, batch_size)
        locs = jnp.zeros(batch_size)
        concentrations = jnp.ones(batch_size) * 5.0
        
        # Without vmap (sequential)
        sequential_times = []
        for _ in range(n_runs):
            start = time.time()
            samples = []
            for i in range(batch_size):
                samples.append(sample_fn(keys[i], locs[i], concentrations[i]))
            sequential_times.append(time.time() - start)
        
        # With vmap
        vmap_times = []
        for _ in range(n_runs):
            start = time.time()
            _ = sample_vmap_jit(keys, locs, concentrations)
            vmap_times.append(time.time() - start)
        
        results.append({
            'batch_size': batch_size,
            'sequential_mean': np.mean(sequential_times),
            'sequential_std': np.std(sequential_times),
            'vmap_mean': np.mean(vmap_times),
            'vmap_std': np.std(vmap_times),
            'speedup': np.mean(sequential_times) / np.mean(vmap_times)
        })
        
        print(f"Batch size {batch_size}: vmap speedup: {results[-1]['speedup']:.2f}x")
    
    return pd.DataFrame(results)


def benchmark_pmap_multi_device(samples_per_device_list, n_runs=3):
    """Benchmark pmap performance on multiple devices."""
    results = []
    
    # Check available devices
    n_devices = jax.device_count()
    if n_devices <= 1:
        print("Multi-device benchmarking requires more than one device")
        return pd.DataFrame()
    
    # Define pmap sampling function with fixed shape for each device
    @partial(pmap, axis_name='devices')
    def sample_pmap(key, loc, concentration, n_samples):
        shape = (n_samples,)
        return sample_von_mises(key, loc, concentration, shape=shape)
    
    # Warmup
    keys = random.split(random.PRNGKey(42), n_devices)
    locs = jnp.ones(n_devices) * jnp.pi/4
    concentrations = jnp.ones(n_devices) * 5.0
    _ = sample_pmap(keys, locs, concentrations, 1000)
    
    for samples_per_device in samples_per_device_list:
        # Single device times
        single_device_times = []
        for _ in range(n_runs):
            start = time.time()
            _ = sample_von_mises(
                random.PRNGKey(0), 
                jnp.pi/4, 
                5.0, 
                shape=(samples_per_device * n_devices,)
            )
            single_device_times.append(time.time() - start)
        
        # Multi-device times
        multi_device_times = []
        for _ in range(n_runs):
            start = time.time()
            _ = sample_pmap(keys, locs, concentrations, samples_per_device)
            multi_device_times.append(time.time() - start)
        
        results.append({
            'samples_per_device': samples_per_device,
            'total_samples': samples_per_device * n_devices,
            'n_devices': n_devices,
            'single_device_mean': np.mean(single_device_times),
            'single_device_std': np.std(single_device_times),
            'multi_device_mean': np.mean(multi_device_times),
            'multi_device_std': np.std(multi_device_times),
            'speedup': np.mean(single_device_times) / np.mean(multi_device_times)
        })
        
        print(f"Multi-device ({n_devices} devices) speedup: {results[-1]['speedup']:.2f}x")
    
    return pd.DataFrame(results)


def benchmark_neural_network(batch_size_list, n_runs=5):
    """Benchmark von Mises layer in neural network context."""
    results = []
    
    # Define a simple test setup for the von Mises layer
    von_mises_jit = jit(von_mises_layer)
    
    # Warmup
    key = random.PRNGKey(42)
    _ = von_mises_jit(
        key,
        jnp.zeros((10, 1)), 
        jnp.ones((10, 1)) * 5.0, 
        temperature=1.0,
        training=True
    )
    
    for batch_size in batch_size_list:
        # Generate random inputs
        mean_logits = jnp.zeros((batch_size, 1))
        concentration = jnp.ones((batch_size, 1)) * 5.0
        
        # Benchmark
        times = []
        for _ in range(n_runs):
            start = time.time()
            _, _ = von_mises_jit(
                key, 
                mean_logits, 
                concentration, 
                temperature=1.0, 
                training=True
            )
            times.append(time.time() - start)
        
        results.append({
            'batch_size': batch_size,
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'samples_per_sec': batch_size / np.mean(times)
        })
        
        print(f"Neural network batch size {batch_size}: {results[-1]['samples_per_sec']:.0f} samples/second")
    
    return pd.DataFrame(results)


def profile_memory_usage():
    """Profile memory usage for different sample sizes."""
    try:
        import psutil
    except ImportError:
        print("psutil not available. Install with 'pip install psutil'")
        return pd.DataFrame()
    
    results = []
    sample_sizes = [10, 100, 1000, 10000, 100000, 500000]
    
    process = psutil.Process(os.getpid())
    key = random.PRNGKey(42)
    
    # Define jitted function to avoid compilation overhead
    sample_jitted = jit(sample_von_mises, static_argnums=(3,))
    
    # Warmup
    _ = sample_jitted(key, 0.0, 5.0, (1000,))
    
    for size in sample_sizes:
        # Measure baseline memory
        baseline = process.memory_info().rss / 1024 / 1024  # in MB
        
        # Sample
        samples = sample_jitted(key, 0.0, 5.0, (size,))
        
        # Measure memory after sampling
        after_sample = process.memory_info().rss / 1024 / 1024  # in MB
        
        # Force JIT compilation to transfer result from GPU if needed
        samples.block_until_ready()
        
        # Measure memory after transfer
        after_transfer = process.memory_info().rss / 1024 / 1024  # in MB
        
        results.append({
            'sample_size': size,
            'baseline_mb': baseline,
            'after_sample_mb': after_sample,
            'after_transfer_mb': after_transfer,
            'sample_overhead_mb': after_sample - baseline,
            'transfer_overhead_mb': after_transfer - after_sample
        })
        
        print(f"Sample size {size}: Memory overhead {results[-1]['sample_overhead_mb']:.2f} MB")
    
    return pd.DataFrame(results)


def plot_benchmark_results(save_dir='benchmark_results'):
    """Run all benchmarks and plot results."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Show available info
    print("\n=== JAX Setup Information ===")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default device: {jax.default_backend()}")
    print("=" * 50)
    
    results = {}
    
    try:
        # JIT vs No JIT
        print("\n=== Benchmarking JIT vs No JIT ===")
        jit_df = benchmark_jit_vs_nojit([100, 1000, 10000])
        results['jit_df'] = jit_df
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(jit_df['n_samples'], jit_df['nojit_mean'], yerr=jit_df['nojit_std'], label='No JIT')
        plt.errorbar(jit_df['n_samples'], jit_df['jit_mean'], yerr=jit_df['jit_std'], label='JIT')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of Samples')
        plt.ylabel('Time (seconds)')
        plt.title('JIT vs No JIT Performance')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(f'{save_dir}/jit_vs_nojit.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error in JIT benchmarking: {e}")
    
    try:
        # Concentration impact
        print("\n=== Benchmarking concentration impact ===")
        concentration_df = benchmark_concentration_impact([0.01, 0.1, 1.0, 5.0, 10.0, 100.0])
        results['concentration_df'] = concentration_df
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(concentration_df['concentration'], concentration_df['time_mean'], 
                    yerr=concentration_df['time_std'])
        plt.xscale('log')
        plt.xlabel('Concentration (κ)')
        plt.ylabel('Time (seconds)')
        plt.title('Impact of Concentration Parameter on Performance')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(f'{save_dir}/concentration_impact.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error in concentration benchmarking: {e}")
    
    try:
        # vmap scaling
        print("\n=== Benchmarking vmap scaling ===")
        vmap_df = benchmark_vmap_scaling([1, 10, 100, 1000])
        results['vmap_df'] = vmap_df
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(vmap_df['batch_size'], vmap_df['sequential_mean'], 
                    yerr=vmap_df['sequential_std'], label='Sequential')
        plt.errorbar(vmap_df['batch_size'], vmap_df['vmap_mean'], 
                    yerr=vmap_df['vmap_std'], label='vmap')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Batch Size')
        plt.ylabel('Time (seconds)')
        plt.title('vmap Performance Scaling')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(f'{save_dir}/vmap_scaling.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error in vmap benchmarking: {e}")
    
    # pmap multi-device (if applicable)
    if jax.device_count() > 1:
        try:
            print("\n=== Benchmarking pmap multi-device ===")
            pmap_df = benchmark_pmap_multi_device([100, 1000, 10000])
            results['pmap_df'] = pmap_df
            
            plt.figure(figsize=(10, 6))
            plt.errorbar(pmap_df['total_samples'], pmap_df['single_device_mean'], 
                        yerr=pmap_df['single_device_std'], label='Single Device')
            plt.errorbar(pmap_df['total_samples'], pmap_df['multi_device_mean'], 
                        yerr=pmap_df['multi_device_std'], label=f'{jax.device_count()} Devices')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Total Number of Samples')
            plt.ylabel('Time (seconds)')
            plt.title('Multi-Device Performance Scaling')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.savefig(f'{save_dir}/pmap_scaling.png', dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Error in pmap benchmarking: {e}")
    
    try:
        # Neural network
        print("\n=== Benchmarking neural network integration ===")
        nn_df = benchmark_neural_network([1, 10, 100, 1000])
        results['nn_df'] = nn_df
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(nn_df['batch_size'], nn_df['time_mean'], yerr=nn_df['time_std'])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Batch Size')
        plt.ylabel('Time (seconds)')
        plt.title('Neural Network Integration Performance')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(f'{save_dir}/neural_network.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error in neural network benchmarking: {e}")
    
    try:
        # Memory usage
        print("\n=== Profiling memory usage ===")
        memory_df = profile_memory_usage()
        if not memory_df.empty:
            results['memory_df'] = memory_df
            
            plt.figure(figsize=(10, 6))
            plt.plot(memory_df['sample_size'], memory_df['sample_overhead_mb'], 'o-', label='Sampling Overhead')
            plt.plot(memory_df['sample_size'], memory_df['transfer_overhead_mb'], 'o-', label='Transfer Overhead')
            plt.xscale('log')
            plt.xlabel('Sample Size')
            plt.ylabel('Memory Overhead (MB)')
            plt.title('Memory Usage Profile')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.savefig(f'{save_dir}/memory_profile.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error in memory profiling: {e}")
    
    # Save all dataframes
    print("\n=== Saving benchmark results ===")
    for name, df in results.items():
        if not df.empty:
            df.to_csv(f'{save_dir}/{name}.csv', index=False)
    
    print(f"Benchmark results saved to {save_dir}")
    
    # Print summary
    print("\n=== Benchmark Summary ===")
    print("=" * 50)
    if 'jit_df' in results:
        print(f"JIT provides {results['jit_df']['speedup'].mean():.1f}x speedup on average")
    if 'vmap_df' in results:
        print(f"vmap provides {results['vmap_df']['speedup'].mean():.1f}x speedup for batched operations")
    if 'pmap_df' in results and jax.device_count() > 1:
        print(f"Multi-device execution with pmap shows {results['pmap_df']['speedup'].mean():.1f}x speedup with {jax.device_count()} devices")
    print("=" * 50)


if __name__ == "__main__":
    # Run all benchmarks
    plot_benchmark_results() 