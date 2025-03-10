#!/usr/bin/env python
"""
Script to compare performance between JAX von Mises and SciPy's vonmises implementation.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from scipy.stats import vonmises

# Set plotting style
plt.style.use('ggplot')
sns.set_context("notebook", font_scale=1.5)

# Import JAX and the von Mises library
import jax
import jax.numpy as jnp
from jax import random
from jax_von_mises import sample_von_mises, vmises_log_prob, vmises_entropy

print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")

# Create directory for figures
import os
os.makedirs('figures', exist_ok=True)

def benchmark_sampling_performance():
    """Compare sampling performance between JAX and SciPy implementations."""
    
    # Prepare JAX sampler with JIT
    jitted_sampler = jax.jit(sample_von_mises, static_argnums=(3,))
    key = random.PRNGKey(42)
    
    # Sample sizes to test
    sample_sizes = [100, 1000, 10000, 100000]
    n_runs = 3  # Number of runs to average
    
    results = []
    
    print("Benchmarking sampling performance...")
    
    # Warmup JAX JIT compilation (to separate compilation time from execution time)
    print("Warming up JAX JIT...")
    _ = jitted_sampler(key, 0.0, 5.0, shape=(10,))
    
    for size in sample_sizes:
        print(f"Testing sample size: {size}")
        
        # Test JAX implementation
        print("  Testing JAX implementation...")
        start_time = time.time()
        for i in range(n_runs):
            subkey = random.fold_in(key, i)
            samples = jitted_sampler(subkey, 0.0, 5.0, shape=(size,))
            # Force completion of any asynchronous operations
            samples.block_until_ready()
        jax_time = time.time() - start_time
        jax_samples_per_sec = (n_runs * size) / jax_time
        
        # Test SciPy implementation
        print("  Testing SciPy implementation...")
        start_time = time.time()
        for i in range(n_runs):
            samples = vonmises.rvs(kappa=5.0, loc=0.0, size=size)
        scipy_time = time.time() - start_time
        scipy_samples_per_sec = (n_runs * size) / scipy_time
        
        # Calculate speedup
        speedup = jax_samples_per_sec / scipy_samples_per_sec
        
        results.append({
            'Sample Size': size,
            'JAX Samples/Second': jax_samples_per_sec,
            'SciPy Samples/Second': scipy_samples_per_sec,
            'Speedup': speedup
        })
        
        print(f"  JAX: {jax_samples_per_sec:.0f} samples/sec")
        print(f"  SciPy: {scipy_samples_per_sec:.0f} samples/sec")
        print(f"  Speedup: {speedup:.1f}x")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    print("\nResults summary:")
    print(df)
    
    # Create bar plot
    plt.figure(figsize=(14, 8))
    
    # Convert to long format for easier plotting
    df_long = pd.melt(df, id_vars=['Sample Size'], 
                       value_vars=['JAX Samples/Second', 'SciPy Samples/Second'],
                       var_name='Implementation', value_name='Samples/Second')
    
    # Replace long names with shorter ones
    df_long['Implementation'] = df_long['Implementation'].replace({
        'JAX Samples/Second': 'JAX von Mises', 
        'SciPy Samples/Second': 'SciPy vonmises'
    })
    
    # Convert sample size to string for categorical plotting
    df_long['Sample Size'] = df_long['Sample Size'].astype(str)
    
    # Create grouped bar plot
    ax = sns.barplot(x='Sample Size', y='Samples/Second', hue='Implementation', 
                    data=df_long, palette=['steelblue', 'lightcoral'])
    
    # Add text labels for speedup
    for i, size in enumerate(sample_sizes):
        jax_val = df[df['Sample Size'] == size]['JAX Samples/Second'].values[0]
        scipy_val = df[df['Sample Size'] == size]['SciPy Samples/Second'].values[0]
        speedup = df[df['Sample Size'] == size]['Speedup'].values[0]
        
        # Place text above the JAX bar
        ax.text(i, jax_val * 1.05, f"{speedup:.1f}x faster", 
                ha='center', fontweight='bold', fontsize=12)
    
    plt.title('Sampling Performance: JAX von Mises vs SciPy vonmises', fontsize=16)
    plt.ylabel('Samples per Second (higher is better)', fontsize=14)
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(axis='y', alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('figures/jax_vs_scipy_sampling.png', dpi=300, bbox_inches='tight')
    print(f"Saved jax_vs_scipy_sampling.png")
    
    return df

def benchmark_pdf_performance():
    """Compare PDF computation performance between JAX and SciPy."""
    
    # Prepare JAX functions with JIT
    jitted_pdf = jax.jit(lambda x, loc, kappa: jnp.exp(vmises_log_prob(x, loc, kappa)))
    
    # Prepare test data
    test_sizes = [1000, 10000, 100000, 1000000]
    n_runs = 3  # Number of runs to average
    
    results = []
    
    print("\nBenchmarking PDF computation performance...")
    
    # Warmup JAX JIT compilation
    x_warmup = jnp.linspace(-jnp.pi, jnp.pi, 10)
    _ = jitted_pdf(x_warmup, 0.0, 5.0)
    
    for size in test_sizes:
        print(f"Testing array size: {size}")
        
        # Generate test data
        x = jnp.linspace(-jnp.pi, jnp.pi, size)
        x_np = np.linspace(-np.pi, np.pi, size)
        
        # Test JAX implementation
        print("  Testing JAX implementation...")
        start_time = time.time()
        for i in range(n_runs):
            pdf = jitted_pdf(x, 0.0, 5.0)
            pdf.block_until_ready()
        jax_time = time.time() - start_time
        jax_values_per_sec = (n_runs * size) / jax_time
        
        # Test SciPy implementation
        print("  Testing SciPy implementation...")
        start_time = time.time()
        for i in range(n_runs):
            pdf = vonmises.pdf(x_np, kappa=5.0, loc=0.0)
        scipy_time = time.time() - start_time
        scipy_values_per_sec = (n_runs * size) / scipy_time
        
        # Calculate speedup
        speedup = jax_values_per_sec / scipy_values_per_sec
        
        results.append({
            'Array Size': size,
            'JAX Values/Second': jax_values_per_sec,
            'SciPy Values/Second': scipy_values_per_sec,
            'Speedup': speedup
        })
        
        print(f"  JAX: {jax_values_per_sec:.0f} values/sec")
        print(f"  SciPy: {scipy_values_per_sec:.0f} values/sec")
        print(f"  Speedup: {speedup:.1f}x")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    print("\nResults summary:")
    print(df)
    
    # Create bar plot
    plt.figure(figsize=(14, 8))
    
    # Convert to long format for easier plotting
    df_long = pd.melt(df, id_vars=['Array Size'], 
                      value_vars=['JAX Values/Second', 'SciPy Values/Second'],
                      var_name='Implementation', value_name='Values/Second')
    
    # Replace long names with shorter ones
    df_long['Implementation'] = df_long['Implementation'].replace({
        'JAX Values/Second': 'JAX von Mises', 
        'SciPy Values/Second': 'SciPy vonmises'
    })
    
    # Convert array size to string for categorical plotting
    df_long['Array Size'] = df_long['Array Size'].astype(str)
    
    # Create grouped bar plot
    ax = sns.barplot(x='Array Size', y='Values/Second', hue='Implementation', 
                    data=df_long, palette=['steelblue', 'lightcoral'])
    
    # Add text labels for speedup
    for i, size in enumerate(test_sizes):
        jax_val = df[df['Array Size'] == size]['JAX Values/Second'].values[0]
        scipy_val = df[df['Array Size'] == size]['SciPy Values/Second'].values[0]
        speedup = df[df['Array Size'] == size]['Speedup'].values[0]
        
        # Place text above the JAX bar
        ax.text(i, jax_val * 1.05, f"{speedup:.1f}x faster", 
                ha='center', fontweight='bold', fontsize=12)
    
    plt.title('PDF Computation Performance: JAX von Mises vs SciPy vonmises', fontsize=16)
    plt.ylabel('Values per Second (higher is better)', fontsize=14)
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(axis='y', alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('figures/jax_vs_scipy_pdf.png', dpi=300, bbox_inches='tight')
    print(f"Saved jax_vs_scipy_pdf.png")
    
    return df

def benchmark_entropy_performance():
    """Compare entropy calculation performance between JAX and SciPy implementations."""
    print("\nBenchmarking entropy calculation performance...")
    
    # Prepare JAX entropy with JIT
    jitted_entropy = jax.jit(vmises_entropy)
    vmapped_entropy = jax.vmap(jitted_entropy)
    
    # Warmup JAX JIT compilation
    print("Warming up JAX JIT...")
    _ = jitted_entropy(1.0)
    _ = vmapped_entropy(jnp.array([1.0, 2.0]))
    
    # Array sizes to test
    test_sizes = [10, 100, 1000, 10000, 100000]
    n_runs = 3  # Number of runs to average
    
    # Prepare results
    results = []
    
    for size in test_sizes:
        print(f"Testing array size: {size}")
        
        # Generate random concentration values
        kappa_values = np.random.uniform(0.1, 10.0, size=size)
        jax_kappa = jnp.array(kappa_values)
        
        # JAX entropy timing (with vmap+JIT)
        jax_times = []
        for _ in range(n_runs):
            start_time = time.time()
            _ = vmapped_entropy(jax_kappa).block_until_ready()
            jax_times.append(time.time() - start_time)
        avg_jax_time = sum(jax_times) / n_runs
        
        # SciPy entropy timing
        scipy_times = []
        for _ in range(n_runs):
            start_time = time.time()
            _ = [scipy.stats.vonmises.entropy(k) for k in kappa_values]
            scipy_times.append(time.time() - start_time)
        avg_scipy_time = sum(scipy_times) / n_runs
        
        # Calculate values per second
        jax_vals_per_sec = size / avg_jax_time
        scipy_vals_per_sec = size / avg_scipy_time
        speedup = jax_vals_per_sec / scipy_vals_per_sec
        
        # Record results
        results.append({
            'Array Size': size,
            'JAX Time (s)': avg_jax_time,
            'SciPy Time (s)': avg_scipy_time,
            'JAX Values/Second': jax_vals_per_sec,
            'SciPy Values/Second': scipy_vals_per_sec,
            'Speedup': speedup
        })
        
        print(f"  JAX: {avg_jax_time:.6f}s ({jax_vals_per_sec:.1f} vals/sec)")
        print(f"  SciPy: {avg_scipy_time:.6f}s ({scipy_vals_per_sec:.1f} vals/sec)")
        print(f"  Speedup: {speedup:.1f}x")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Reshape data for seaborn
    df_long = pd.melt(df, id_vars=['Array Size'], 
                      value_vars=['JAX Values/Second', 'SciPy Values/Second'],
                      var_name='Implementation', value_name='Values/Second')
    
    # Replace long names with shorter ones
    df_long['Implementation'] = df_long['Implementation'].replace({
        'JAX Values/Second': 'JAX von Mises', 
        'SciPy Values/Second': 'SciPy vonmises'
    })
    
    # Convert array size to string for categorical plotting
    df_long['Array Size'] = df_long['Array Size'].astype(str)
    
    # Create grouped bar plot
    ax = sns.barplot(x='Array Size', y='Values/Second', hue='Implementation', 
                    data=df_long, palette=['steelblue', 'lightcoral'])
    
    # Add text labels for speedup
    for i, size in enumerate(test_sizes):
        jax_val = df[df['Array Size'] == size]['JAX Values/Second'].values[0]
        scipy_val = df[df['Array Size'] == size]['SciPy Values/Second'].values[0]
        speedup = df[df['Array Size'] == size]['Speedup'].values[0]
        
        # Place text above the JAX bar
        ax.text(i, jax_val * 1.05, f"{speedup:.1f}x faster", 
                ha='center', fontweight='bold', fontsize=12)
    
    plt.title('Entropy Calculation Performance: JAX von Mises vs SciPy vonmises', fontsize=16)
    plt.ylabel('Calculations per Second (higher is better)', fontsize=14)
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(axis='y', alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('figures/entropy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved entropy_comparison.png")
    
    return df

def create_feature_comparison_table():
    """Create a visual table comparing features between JAX and SciPy implementations."""
    
    # Define features and their support status
    features = [
        'Random sampling',
        'PDF computation',
        'GPU/TPU acceleration',
        'JIT compilation',
        'Batch processing',
        'Neural network integration',
        'Gradient computation',
        'CDF computation',
        'PPF (inverse CDF)',
        'Parameter fitting',
        'Moment calculations',
        'Entropy calculation'
    ]
    
    jax_support = [
        'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes',
        'No', 'No', 'No', 'No', 'Yes'
    ]
    
    scipy_support = [
        'Yes', 'Yes', 'No', 'No', 'Limited', 'No', 'No',
        'Yes', 'Yes', 'Yes', 'Yes', 'Yes'
    ]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Feature': features,
        'JAX von Mises': jax_support,
        'SciPy vonmises': scipy_support
    })
    
    # Create a figure
    fig, ax = plt.figure(figsize=(12, 10)), plt.subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    # Create table with cell colors
    cell_colors = []
    for row in df.values:
        row_colors = []
        for i, val in enumerate(row):
            if i == 0:  # Feature column
                row_colors.append('#f0f0f0')
            else:  # Implementation columns
                if val == 'Yes':
                    row_colors.append('#d0e9ff')
                elif val == 'No':
                    row_colors.append('#ffe6e6')
                else:
                    row_colors.append('#fff9d0')
        cell_colors.append(row_colors)
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        cellColours=cell_colors,
        colColours=['#e0e0e0', '#a8d1ff', '#ffcccc']  # Header colors
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Add title
    plt.title('Feature Comparison: JAX von Mises vs SciPy vonmises', fontsize=16, pad=20)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('figures/feature_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved feature_comparison.png")

def create_visualization_comparison():
    """Create a side-by-side visual comparison of both implementations."""
    
    try:
        print("Starting visualization comparison...")
        
        # Create just a simple cartesian plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        print("Created figure and axes")
        
        # Parameters
        loc = 0.0
        kappa = 5.0
        sample_size = 10000
        print(f"Using parameters: loc={loc}, kappa={kappa}, sample_size={sample_size}")
        
        # Generate samples from both implementations
        print("Generating JAX samples...")
        key = random.PRNGKey(42)
        jax_samples = sample_von_mises(key, loc, kappa, shape=(sample_size,))
        
        print("Generating SciPy samples...")
        scipy_samples = vonmises.rvs(kappa=kappa, loc=loc, size=sample_size)
        print("Samples generated successfully")
        
        # Calculate PDF values
        print("Calculating PDF values...")
        x = np.linspace(-np.pi, np.pi, 1000)
        jax_pdf = np.exp(vmises_log_prob(x, loc, kappa))
        scipy_pdf = vonmises.pdf(x, kappa=kappa, loc=loc)
        print("PDF values calculated successfully")
        
        # JAX plot
        print("Creating JAX plot...")
        axes[0].hist(jax_samples, bins=50, density=True, alpha=0.6, color='steelblue', label='Samples')
        axes[0].plot(x, jax_pdf, 'b-', linewidth=2, label='PDF')
        axes[0].set_title('JAX von Mises Distribution', fontsize=14)
        axes[0].set_xlabel('Angle (radians)', fontsize=12)
        axes[0].set_ylabel('Density', fontsize=12)
        axes[0].set_xlim(-np.pi, np.pi)
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        print("JAX plot created successfully")
        
        # SciPy plot
        print("Creating SciPy plot...")
        axes[1].hist(scipy_samples, bins=50, density=True, alpha=0.6, color='lightcoral', label='Samples')
        axes[1].plot(x, scipy_pdf, 'r-', linewidth=2, label='PDF')
        axes[1].set_title('SciPy vonmises Distribution', fontsize=14)
        axes[1].set_xlabel('Angle (radians)', fontsize=12)
        axes[1].set_ylabel('Density', fontsize=12)
        axes[1].set_xlim(-np.pi, np.pi)
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        print("SciPy plot created successfully")
        
        print("Saving visualization comparison...")
        plt.tight_layout()
        plt.savefig('figures/visualization_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved visualization_comparison.png")
        
        print("Creating overlay comparison...")
        # Create a separate image with both distributions overlaid for direct comparison
        plt.figure(figsize=(12, 6))
        plt.hist(jax_samples, bins=50, density=True, alpha=0.3, color='steelblue', label='JAX Samples')
        plt.hist(scipy_samples, bins=50, density=True, alpha=0.3, color='lightcoral', label='SciPy Samples')
        plt.plot(x, jax_pdf, 'b-', linewidth=2, label='JAX PDF')
        plt.plot(x, scipy_pdf, 'r--', linewidth=2, label='SciPy PDF')
        plt.title('JAX vs SciPy von Mises Distribution Comparison', fontsize=14)
        plt.xlabel('Angle (radians)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.xlim(-np.pi, np.pi)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('figures/distribution_overlay.png', dpi=300, bbox_inches='tight')
        print(f"Saved distribution_overlay.png")
        
        return True
    except Exception as e:
        print(f"Error in visualization comparison: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Generating SciPy comparison visualizations...")
    
    # Run benchmarks
    sampling_df = benchmark_sampling_performance()
    pdf_df = benchmark_pdf_performance()
    entropy_df = benchmark_entropy_performance()
    
    # Create summary visualizations
    create_feature_comparison_table()
    create_visualization_comparison()
    
    print("All SciPy comparison visualizations complete!") 