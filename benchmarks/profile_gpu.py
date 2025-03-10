"""
GPU-specific performance profiling for JAX von Mises sampling.
This script performs detailed GPU profiling of the von Mises sampling functions.
"""

import time
import os
import json
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from jax import random, jit, vmap, pmap, lax
try:
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.initialize_cache("./jax_cache")
except ImportError:
    print("JAX compilation cache not available in this version")

from jax_von_mises.sampler import sample_von_mises, vmises_log_prob, compute_p

def profile_gpu_performance():
    """Comprehensive GPU performance profiler."""
    
    # Check for GPU availability
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == 'gpu']
    
    print("=== JAX von Mises GPU Profiling ===")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {devices}")
    
    if not gpu_devices:
        print("\nWARNING: No GPU devices detected. Running on CPU only.")
        print("To enable GPU support, please run the install_gpu.sh script.")
    else:
        print(f"\nFound {len(gpu_devices)} GPU device(s):")
        for i, device in enumerate(gpu_devices):
            print(f"  GPU {i}: {device}")
    
    # Create a dictionary to store all timing results
    results = {
        "environment": {
            "jax_version": jax.__version__,
            "device_count": len(devices),
            "gpu_count": len(gpu_devices),
            "devices": [str(d) for d in devices],
            "xla_flags": os.environ.get("XLA_FLAGS", "")
        },
        "function_timings": {},
        "concentration_scaling": {},
        "batch_size_scaling": {},
        "sample_size_scaling": {},
        "multi_device_scaling": {}
    }
    
    # Set fixed random seed for reproducibility
    key = random.PRNGKey(42)
    
    # Create test parameters
    loc = 0.0
    concentration = 5.0
    sample_size = 10000
    
    # 1. Basic function timing
    print("\n--- Basic Function Timing ---")
    
    # Profile vmises_log_prob
    print("\nProfiling vmises_log_prob...")
    
    # JIT compile for accurate timing
    vmises_log_prob_jit = jit(vmises_log_prob)
    # Warmup
    _ = vmises_log_prob_jit(0.0, loc, concentration)
    
    # Time execution
    iterations = 1000
    start_time = time.time()
    for _ in range(iterations):
        _ = vmises_log_prob_jit(0.0, loc, concentration)
    jax.block_until_ready(_)
    elapsed = time.time() - start_time
    
    per_call_ms = (elapsed / iterations) * 1000
    print(f"  Time: {elapsed:.4f}s for {iterations} iterations ({per_call_ms:.4f}ms per call)")
    
    results["function_timings"]["vmises_log_prob"] = {
        "total_time": elapsed,
        "iterations": iterations,
        "time_per_call_ms": per_call_ms
    }
    
    # Profile compute_p
    print("\nProfiling compute_p...")
    
    # JIT compile for accurate timing
    compute_p_jit = jit(compute_p)
    # Warmup
    _ = compute_p_jit(concentration)
    
    # Time execution
    start_time = time.time()
    for _ in range(iterations):
        _ = compute_p_jit(concentration)
    jax.block_until_ready(_)
    elapsed = time.time() - start_time
    
    per_call_ms = (elapsed / iterations) * 1000
    print(f"  Time: {elapsed:.4f}s for {iterations} iterations ({per_call_ms:.4f}ms per call)")
    
    results["function_timings"]["compute_p"] = {
        "total_time": elapsed,
        "iterations": iterations,
        "time_per_call_ms": per_call_ms
    }
    
    # 2. JIT vs. non-JIT sampling
    print("\n--- JIT vs. Non-JIT Sampling ---")
    
    sample_iterations = 20
    
    # JIT-compiled version with static_argnums for shape parameter
    sample_jit = jit(sample_von_mises, static_argnums=(3,))
    # Warmup
    _ = sample_jit(key, loc, concentration, shape=(sample_size,))
    
    # Measure JIT performance
    print("\nProfiling sample_von_mises with JIT...")
    start_time = time.time()
    for i in range(sample_iterations):
        subkey = random.fold_in(key, i)
        result = sample_jit(subkey, loc, concentration, shape=(sample_size,))
    jax.block_until_ready(result)
    jit_elapsed = time.time() - start_time
    
    per_call_ms_jit = (jit_elapsed / sample_iterations) * 1000
    samples_per_sec_jit = (sample_size * sample_iterations) / jit_elapsed
    
    print(f"  Time: {jit_elapsed:.4f}s for {sample_iterations} iterations ({per_call_ms_jit:.2f}ms per call)")
    print(f"  Samples per second: {samples_per_sec_jit:.0f}")
    
    # Reduce iterations for non-JIT version as it's much slower
    non_jit_iterations = 3
    
    # Measure non-JIT performance
    print("\nProfiling sample_von_mises without JIT...")
    start_time = time.time()
    for i in range(non_jit_iterations):
        subkey = random.fold_in(key, i)
        result = sample_von_mises(subkey, loc, concentration, shape=(sample_size,))
    jax.block_until_ready(result)
    non_jit_elapsed = time.time() - start_time
    
    per_call_ms_non_jit = (non_jit_elapsed / non_jit_iterations) * 1000
    samples_per_sec_non_jit = (sample_size * non_jit_iterations) / non_jit_elapsed
    
    print(f"  Time: {non_jit_elapsed:.4f}s for {non_jit_iterations} iterations ({per_call_ms_non_jit:.2f}ms per call)")
    print(f"  Samples per second: {samples_per_sec_non_jit:.0f}")
    
    # Calculate speedup
    speedup = per_call_ms_non_jit / per_call_ms_jit
    print(f"  JIT speedup: {speedup:.1f}x")
    
    results["function_timings"]["sample_von_mises_jit"] = {
        "total_time": jit_elapsed,
        "iterations": sample_iterations,
        "time_per_call_ms": per_call_ms_jit,
        "samples_per_second": samples_per_sec_jit
    }
    
    results["function_timings"]["sample_von_mises_non_jit"] = {
        "total_time": non_jit_elapsed,
        "iterations": non_jit_iterations,
        "time_per_call_ms": per_call_ms_non_jit,
        "samples_per_second": samples_per_sec_non_jit,
        "jit_speedup": speedup
    }
    
    # 3. Concentration parameter impact
    print("\n--- Concentration Parameter Impact ---")
    
    concentration_values = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 1000.0]
    concentration_timings = []
    
    for kappa in concentration_values:
        print(f"\nTesting with κ={kappa}...")
        
        # Warmup
        _ = sample_jit(key, loc, kappa, shape=(sample_size,))
        
        start_time = time.time()
        for i in range(sample_iterations):
            subkey = random.fold_in(key, i)
            result = sample_jit(subkey, loc, kappa, shape=(sample_size,))
        jax.block_until_ready(result)
        elapsed = time.time() - start_time
        
        per_call_ms = (elapsed / sample_iterations) * 1000
        samples_per_sec = (sample_size * sample_iterations) / elapsed
        
        print(f"  Time: {elapsed:.4f}s for {sample_iterations} iterations ({per_call_ms:.2f}ms per call)")
        print(f"  Samples per second: {samples_per_sec:.0f}")
        
        concentration_timings.append({
            "kappa": kappa,
            "total_time": elapsed,
            "time_per_call_ms": per_call_ms,
            "samples_per_second": samples_per_sec
        })
    
    results["concentration_scaling"] = concentration_timings
    
    # 4. Batch processing with vmap
    print("\n--- Batch Processing with vmap ---")
    
    batch_sizes = [1, 10, 100, 1000]
    batch_timings = []
    
    # Define a function to sample
    def sample_fn(key, loc, concentration):
        return sample_von_mises(key, loc, concentration, shape=(100,))
    
    # Create JIT+VMAP version
    jit_vmap_sample = jit(vmap(sample_fn, in_axes=(0, 0, 0)))
    
    for batch_size in batch_sizes:
        print(f"\nTesting with batch size {batch_size}...")
        
        # Create batch parameters
        batch_keys = random.split(key, batch_size)
        batch_locs = jnp.ones(batch_size) * loc
        batch_concentrations = jnp.ones(batch_size) * concentration
        
        # Warmup
        _ = jit_vmap_sample(batch_keys, batch_locs, batch_concentrations)
        
        # Time execution
        start_time = time.time()
        for i in range(10):
            fold_key = random.fold_in(key, i)
            batch_keys = random.split(fold_key, batch_size)
            result = jit_vmap_sample(batch_keys, batch_locs, batch_concentrations)
        jax.block_until_ready(result)
        elapsed = time.time() - start_time
        
        total_samples = batch_size * 100 * 10  # batch_size * samples_per_batch * iterations
        samples_per_sec = total_samples / elapsed
        
        print(f"  Time: {elapsed:.4f}s for 10 iterations")
        print(f"  Samples per second: {samples_per_sec:.0f}")
        
        # Compare to sequential for small batch sizes
        if batch_size <= 100:
            # Sequential version
            start_time = time.time()
            for i in range(10):
                fold_key = random.fold_in(key, i)
                for j in range(batch_size):
                    subkey = random.fold_in(fold_key, j)
                    result = sample_jit(subkey, loc, concentration, shape=(100,))
            jax.block_until_ready(result)
            seq_elapsed = time.time() - start_time
            
            seq_samples_per_sec = total_samples / seq_elapsed
            speedup = seq_elapsed / elapsed
            
            print(f"  Sequential samples per second: {seq_samples_per_sec:.0f}")
            print(f"  Speedup from vectorization: {speedup:.1f}x")
        else:
            speedup = None
            seq_samples_per_sec = None
        
        batch_timings.append({
            "batch_size": batch_size,
            "total_time": elapsed,
            "samples_per_second": samples_per_sec,
            "sequential_samples_per_second": seq_samples_per_sec,
            "speedup": speedup
        })
    
    results["batch_size_scaling"] = batch_timings
    
    # 5. Sample size scaling
    print("\n--- Sample Size Scaling ---")
    
    sample_sizes = [10, 100, 1000, 10000, 100000, 1000000]
    sample_size_timings = []
    
    for size in sample_sizes:
        print(f"\nTesting with sample size {size}...")
        
        # Warmup
        _ = sample_jit(key, loc, concentration, shape=(size,))
        
        # Time execution
        iter_count = max(1, int(100000 / size))  # Adjust iterations based on size
        start_time = time.time()
        for i in range(iter_count):
            subkey = random.fold_in(key, i)
            result = sample_jit(subkey, loc, concentration, shape=(size,))
        jax.block_until_ready(result)
        elapsed = time.time() - start_time
        
        total_samples = size * iter_count
        samples_per_sec = total_samples / elapsed
        
        print(f"  Time: {elapsed:.4f}s for {iter_count} iterations")
        print(f"  Samples per second: {samples_per_sec:.0f}")
        
        sample_size_timings.append({
            "sample_size": size,
            "iterations": iter_count,
            "total_time": elapsed,
            "samples_per_second": samples_per_sec
        })
    
    results["sample_size_scaling"] = sample_size_timings
    
    # 6. Multi-device execution (if available)
    if len(gpu_devices) > 1:
        print("\n--- Multi-Device Execution ---")
        
        # Define a pmap-compatible function
        @partial(pmap, axis_name='devices')
        def sample_pmap(key, loc, concentration, size):
            return sample_von_mises(key, loc, concentration, shape=(size,))
        
        device_counts = list(range(1, len(gpu_devices) + 1))
        pmap_timings = []
        
        for n_devices in device_counts:
            print(f"\nTesting with {n_devices} devices...")
            
            # Create keys and parameters for each device
            pmap_keys = random.split(key, n_devices)
            pmap_locs = jnp.ones(n_devices) * loc
            pmap_concentrations = jnp.ones(n_devices) * concentration
            
            # Samples per device
            samples_per_device = 100000
            
            # Warmup
            _ = sample_pmap(pmap_keys[:n_devices], 
                            pmap_locs[:n_devices], 
                            pmap_concentrations[:n_devices],
                            samples_per_device)
            
            # Time execution
            iterations = 10
            start_time = time.time()
            for i in range(iterations):
                fold_key = random.fold_in(key, i)
                device_keys = random.split(fold_key, n_devices)
                result = sample_pmap(device_keys, 
                                    pmap_locs[:n_devices], 
                                    pmap_concentrations[:n_devices],
                                    samples_per_device)
            jax.block_until_ready(result)
            elapsed = time.time() - start_time
            
            total_samples = samples_per_device * n_devices * iterations
            samples_per_sec = total_samples / elapsed
            
            print(f"  Time: {elapsed:.4f}s for {iterations} iterations")
            print(f"  Total samples per second: {samples_per_sec:.0f}")
            print(f"  Samples per second per device: {samples_per_sec/n_devices:.0f}")
            
            pmap_timings.append({
                "device_count": n_devices,
                "iterations": iterations,
                "total_time": elapsed,
                "samples_per_second": samples_per_sec,
                "samples_per_second_per_device": samples_per_sec/n_devices
            })
        
        results["multi_device_scaling"] = pmap_timings
    
    # Save results
    os.makedirs("profile_results", exist_ok=True)
    with open("profile_results/gpu_profile_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    generate_performance_plots(results)
    
    print("\nProfiling complete. Results saved to profile_results/")
    return results

def generate_performance_plots(results):
    """Generate plots from profiling results."""
    os.makedirs("profile_results", exist_ok=True)
    
    # Set style
    plt.style.use('ggplot')
    
    # 1. Concentration parameter impact
    if results.get("concentration_scaling"):
        concentration_data = results["concentration_scaling"]
        kappas = [item["kappa"] for item in concentration_data]
        samples_per_sec = [item["samples_per_second"] for item in concentration_data]
        
        plt.figure(figsize=(10, 6))
        plt.semilogx(kappas, samples_per_sec, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Concentration Parameter (κ)')
        plt.ylabel('Samples per Second')
        plt.title('Impact of Concentration Parameter on Sampling Performance')
        plt.grid(True, which="both", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig("profile_results/concentration_impact.png", dpi=200)
    
    # 2. Batch size scaling
    if results.get("batch_size_scaling"):
        batch_data = results["batch_size_scaling"]
        batch_sizes = [item["batch_size"] for item in batch_data]
        samples_per_sec = [item["samples_per_second"] for item in batch_data]
        
        plt.figure(figsize=(10, 6))
        plt.loglog(batch_sizes, samples_per_sec, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Batch Size')
        plt.ylabel('Samples per Second')
        plt.title('Scaling with Batch Size (vmap)')
        plt.grid(True, which="both", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig("profile_results/batch_scaling.png", dpi=200)
        
        # Speedup from vectorization
        speedups = [item["speedup"] for item in batch_data if item["speedup"] is not None]
        speedup_sizes = [item["batch_size"] for item in batch_data if item["speedup"] is not None]
        
        if speedups:
            plt.figure(figsize=(10, 6))
            plt.semilogx(speedup_sizes, speedups, 'o-', linewidth=2, markersize=8)
            plt.xlabel('Batch Size')
            plt.ylabel('Speedup Factor')
            plt.title('Vectorization Speedup vs. Sequential Processing')
            plt.grid(True, which="both", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig("profile_results/vectorization_speedup.png", dpi=200)
    
    # 3. Sample size scaling
    if results.get("sample_size_scaling"):
        sample_data = results["sample_size_scaling"]
        sample_sizes = [item["sample_size"] for item in sample_data]
        samples_per_sec = [item["samples_per_second"] for item in sample_data]
        
        plt.figure(figsize=(10, 6))
        plt.loglog(sample_sizes, samples_per_sec, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Sample Size')
        plt.ylabel('Samples per Second')
        plt.title('Performance Scaling with Sample Size')
        plt.grid(True, which="both", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig("profile_results/sample_size_scaling.png", dpi=200)
    
    # 4. Multi-device scaling
    if results.get("multi_device_scaling"):
        pmap_data = results["multi_device_scaling"]
        device_counts = [item["device_count"] for item in pmap_data]
        samples_per_sec = [item["samples_per_second"] for item in pmap_data]
        samples_per_sec_per_device = [item["samples_per_second_per_device"] for item in pmap_data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.plot(device_counts, samples_per_sec, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Devices')
        ax1.set_ylabel('Total Samples per Second')
        ax1.set_title('Multi-Device Scaling (Total Throughput)')
        ax1.grid(True, linestyle="--", alpha=0.7)
        
        ax2.plot(device_counts, samples_per_sec_per_device, 'o-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Devices')
        ax2.set_ylabel('Samples per Second per Device')
        ax2.set_title('Multi-Device Efficiency')
        ax2.grid(True, linestyle="--", alpha=0.7)
        
        plt.tight_layout()
        plt.savefig("profile_results/multi_device_scaling.png", dpi=200)
    
    # 5. JIT vs non-JIT (bar chart)
    if (results.get("function_timings") and 
        "sample_von_mises_jit" in results["function_timings"] and 
        "sample_von_mises_non_jit" in results["function_timings"]):
        
        jit_data = results["function_timings"]["sample_von_mises_jit"]
        non_jit_data = results["function_timings"]["sample_von_mises_non_jit"]
        
        jit_samples_per_sec = jit_data["samples_per_second"]
        non_jit_samples_per_sec = non_jit_data["samples_per_second"]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(['Without JIT', 'With JIT'], 
                     [non_jit_samples_per_sec, jit_samples_per_sec])
        
        # Add speedup annotation
        speedup = jit_samples_per_sec / non_jit_samples_per_sec
        plt.text(1, jit_samples_per_sec * 0.9, f"{speedup:.1f}x faster", 
                ha='center', va='center', fontsize=12)
        
        plt.ylabel('Samples per Second (log scale)')
        plt.title('Impact of JIT Compilation on Sampling Performance')
        plt.yscale('log')
        plt.grid(True, axis='y', linestyle="--", alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig("profile_results/jit_vs_nojit.png", dpi=200)
    
    # 6. Function comparison
    if results.get("function_timings"):
        function_timings = results["function_timings"]
        
        # Extract only basic function timings (not the sampling ones)
        func_times = {k: v["time_per_call_ms"] 
                      for k, v in function_timings.items() 
                      if k not in ["sample_von_mises_jit", "sample_von_mises_non_jit"]}
        
        if func_times:
            plt.figure(figsize=(8, 6))
            plt.bar(func_times.keys(), func_times.values())
            plt.ylabel('Time per Call (ms)')
            plt.title('Function Execution Time Comparison')
            plt.grid(True, axis='y', linestyle="--", alpha=0.7)
            
            # Add value labels
            for i, (func, time) in enumerate(func_times.items()):
                plt.text(i, time, f'{time:.4f} ms', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig("profile_results/function_timings.png", dpi=200)

if __name__ == "__main__":
    profile_gpu_performance() 