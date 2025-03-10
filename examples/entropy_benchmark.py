#!/usr/bin/env python
"""
Script to benchmark entropy calculation performance in the JAX von Mises library.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats

print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")

try:
    from jax_von_mises import vmises_entropy
    print("Successfully imported vmises_entropy function")
except ImportError:
    print("Warning: jax_von_mises package not found. Installing in development mode...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
    from jax_von_mises import vmises_entropy
    print("Successfully installed and imported vmises_entropy function")

def measure_performance():
    """Measure entropy calculation performance."""
    print("\nBenchmarking entropy calculation performance...\n")
    
    # Prepare JAX transformations
    jitted_entropy = jax.jit(vmises_entropy)
    vmapped_entropy = jax.vmap(jitted_entropy)
    
    # Warmup JIT compilation
    print("Warming up JIT compilation...")
    _ = jitted_entropy(1.0)
    _ = vmapped_entropy(jnp.array([1.0]))
    
    # Test sizes
    sizes = [10, 100, 1000, 10000, 100000]
    results = []
    
    for size in sizes:
        print(f"\nTesting array size: {size}")
        
        # Generate random concentration values
        kappa_values = np.random.uniform(0.1, 10.0, size=size)
        jax_kappa = jnp.array(kappa_values)
        
        # SciPy timing
        print("  Measuring SciPy performance...")
        start = time.time()
        _ = [scipy.stats.vonmises.entropy(k) for k in kappa_values]
        scipy_time = time.time() - start
        scipy_rate = size / scipy_time
        
        # Non-JIT timing
        print("  Measuring JAX non-JIT performance...")
        start = time.time()
        for k in kappa_values:
            _ = vmises_entropy(k)
        nonjit_time = time.time() - start
        nonjit_rate = size / nonjit_time
        
        # JIT timing
        print("  Measuring JAX JIT performance...")
        start = time.time()
        for k in kappa_values:
            _ = jitted_entropy(k).block_until_ready()
        jit_time = time.time() - start
        jit_rate = size / jit_time
        
        # vmap timing
        print("  Measuring JAX vmap+JIT performance...")
        start = time.time()
        _ = vmapped_entropy(jax_kappa).block_until_ready()
        vmap_time = time.time() - start
        vmap_rate = size / vmap_time
        
        # Store results
        results.append({
            'Size': size,
            'SciPy (s)': scipy_time,
            'JAX Non-JIT (s)': nonjit_time,
            'JAX JIT (s)': jit_time,
            'JAX vmap+JIT (s)': vmap_time,
            'SciPy (vals/sec)': scipy_rate,
            'JAX Non-JIT (vals/sec)': nonjit_rate,
            'JAX JIT (vals/sec)': jit_rate,
            'JAX vmap+JIT (vals/sec)': vmap_rate,
            'JAX JIT/SciPy Speedup': jit_rate / scipy_rate,
            'JAX vmap/SciPy Speedup': vmap_rate / scipy_rate
        })
        
        # Print results
        print(f"  SciPy: {scipy_time:.6f}s ({scipy_rate:.1f} vals/sec)")
        print(f"  JAX Non-JIT: {nonjit_time:.6f}s ({nonjit_rate:.1f} vals/sec)")
        print(f"  JAX JIT: {jit_time:.6f}s ({jit_rate:.1f} vals/sec)")
        print(f"  JAX vmap+JIT: {vmap_time:.6f}s ({vmap_rate:.1f} vals/sec)")
        print(f"  JAX JIT Speedup over SciPy: {jit_rate / scipy_rate:.1f}x")
        print(f"  JAX vmap+JIT Speedup over SciPy: {vmap_rate / scipy_rate:.1f}x")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    return df

def plot_results(df):
    """Plot benchmark results."""
    plt.figure(figsize=(12, 8))
    
    # Plot values per second
    plt.subplot(2, 1, 1)
    plt.loglog(df['Size'], df['SciPy (vals/sec)'], 'r-o', label='SciPy')
    plt.loglog(df['Size'], df['JAX Non-JIT (vals/sec)'], 'g--s', label='JAX Non-JIT')
    plt.loglog(df['Size'], df['JAX JIT (vals/sec)'], 'b-^', label='JAX JIT')
    plt.loglog(df['Size'], df['JAX vmap+JIT (vals/sec)'], 'm-*', label='JAX vmap+JIT')
    plt.xlabel('Array Size')
    plt.ylabel('Values per Second (log scale)')
    plt.title('Entropy Calculation Performance')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    
    # Plot speedup
    plt.subplot(2, 1, 2)
    plt.semilogx(df['Size'], df['JAX JIT/SciPy Speedup'], 'b-^', label='JAX JIT/SciPy')
    plt.semilogx(df['Size'], df['JAX vmap/SciPy Speedup'], 'm-*', label='JAX vmap+JIT/SciPy')
    plt.axhline(y=1, color='k', linestyle='--')
    plt.xlabel('Array Size')
    plt.ylabel('Speedup over SciPy (x times)')
    plt.title('JAX Speedup over SciPy')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('figures/entropy_benchmark_results.png', dpi=300, bbox_inches='tight')
    print("\nSaved benchmark results to 'figures/entropy_benchmark_results.png'")
    
    return plt

def print_summary(df):
    """Print a summary of the results."""
    print("\nPerformance Summary:")
    print("===================")
    print("\nExecution Time (seconds):")
    print(df[['Size', 'SciPy (s)', 'JAX Non-JIT (s)', 'JAX JIT (s)', 'JAX vmap+JIT (s)']].to_string(index=False))
    
    print("\nSpeed (values per second):")
    print(df[['Size', 'SciPy (vals/sec)', 'JAX Non-JIT (vals/sec)', 'JAX JIT (vals/sec)', 'JAX vmap+JIT (vals/sec)']].to_string(index=False))
    
    print("\nSpeedup over SciPy:")
    print(df[['Size', 'JAX JIT/SciPy Speedup', 'JAX vmap/SciPy Speedup']].to_string(index=False))
    
    # Analysis for different size ranges
    small = df[df['Size'] <= 100]
    medium = df[(df['Size'] > 100) & (df['Size'] <= 1000)]
    large = df[df['Size'] > 1000]
    
    print("\nKey Observations:")
    if not small.empty:
        print(f"- Small arrays (<= 100): SciPy is {small['SciPy (vals/sec)'].mean() / small['JAX vmap+JIT (vals/sec)'].mean():.1f}x faster than JAX vmap+JIT on average")
    
    if not medium.empty:
        comp = medium['JAX vmap+JIT (vals/sec)'].mean() / medium['SciPy (vals/sec)'].mean()
        if comp > 1:
            print(f"- Medium arrays (101-1000): JAX vmap+JIT is {comp:.1f}x faster than SciPy on average")
        else:
            print(f"- Medium arrays (101-1000): SciPy is {1/comp:.1f}x faster than JAX vmap+JIT on average")
    
    if not large.empty:
        print(f"- Large arrays (>1000): JAX vmap+JIT is {large['JAX vmap/SciPy Speedup'].mean():.1f}x faster than SciPy on average")
    
    best_size = df.loc[df['JAX vmap/SciPy Speedup'].idxmax(), 'Size']
    best_speedup = df['JAX vmap/SciPy Speedup'].max()
    print(f"- Maximum speedup: {best_speedup:.1f}x faster with JAX vmap+JIT at size {best_size}")

if __name__ == "__main__":
    print("Starting entropy performance benchmark...")
    results_df = measure_performance()
    plot = plot_results(results_df)
    print_summary(results_df)
    # plt.show()  # Comment out interactive display
    print("\nBenchmark complete.") 