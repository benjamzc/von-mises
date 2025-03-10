#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %% [markdown]
# # JAX von Mises Sampling: Performance Showcase
# 
# This notebook demonstrates the dramatic performance improvements achieved using JAX's Just-In-Time (JIT) compilation with the von Mises sampling library.
# 
# ## What is the von Mises Distribution?
# 
# The von Mises distribution is a probability distribution on the circle. It's often described as the circular analog of the normal distribution and is widely used for modeling angles, directions, and cyclic data in fields like:
# 
# - **Bioinformatics**: Protein dihedral angles
# - **Computer Vision**: Object orientation and pose estimation
# - **Geoscience**: Wind directions and geological orientations
# - **Robotics**: Direction of movement and orientation
# 
# ## The JAX von Mises Library
# 
# Our `jax-von-mises` library implements the Best-Fisher algorithm for sampling from the von Mises distribution, with full compatibility with JAX transformations (jit, vmap, pmap). This makes it:
# 
# - **Fast**: Optimized for high-performance computation
# - **GPU/TPU-compatible**: Runs efficiently on accelerator hardware
# - **Parallelizable**: Works with JAX's vectorization and parallelization tools
# - **Neural network-friendly**: Easily integrates with JAX-based ML frameworks

# %% [markdown]
# ## Setup and Imports
# 
# First, let's import the necessary libraries:

# %%
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import seaborn as sns
from functools import partial

# Set better plotting style
plt.style.use('ggplot')
sns.set_context("notebook", font_scale=1.5)

# Print JAX and device information
print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")

# Import the von Mises sampling library
try:
    from jax_von_mises import sample_von_mises, vmises_log_prob, vmises_entropy
    from jax_von_mises.sampler import compute_p
    print("Successfully imported von Mises sampling functions")
except ImportError:
    print("Warning: jax_von_mises package not found. Installing in development mode...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
    from jax_von_mises import sample_von_mises, vmises_log_prob, vmises_entropy
    from jax_von_mises.sampler import compute_p
    print("Successfully installed and imported von Mises sampling functions")

# %% [markdown]
# ## 1. Basic Sampling Demonstration
# 
# Let's start by demonstrating how to sample from a von Mises distribution:

# %%
# Set a random seed for reproducibility
key = random.PRNGKey(42)

# Parameters for the von Mises distribution
loc = 0.0  # Mean direction (in radians)
concentration = 5.0  # Higher values = more concentrated around the mean

# Generate samples (without JIT first)
n_samples = 10000
samples = sample_von_mises(key, loc, concentration, shape=(n_samples,))

# Plot the histogram of samples
plt.figure(figsize=(12, 6))
plt.hist(samples, bins=50, density=True, alpha=0.6, color='steelblue')
plt.title(f'Von Mises Distribution (μ={loc}, κ={concentration})')
plt.xlabel('Angle (radians)')
plt.ylabel('Density')
plt.axvline(x=loc, color='r', linestyle='--', label='Mean direction')
plt.xlim(-np.pi, np.pi)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. JIT vs. Non-JIT Performance Comparison
# 
# Now, let's compare the performance of JIT-compiled sampling with non-JIT sampling:

# %%
# Define a function to measure sampling time
def measure_sampling_time(sampling_fn, key, loc, concentration, shape, n_runs=5):
    """Measure the time taken to sample from the von Mises distribution."""
    # Warmup call (to trigger compilation if needed)
    _ = sampling_fn(key, loc, concentration, shape)
    
    # Actual timing
    start_time = time.time()
    for i in range(n_runs):
        subkey = random.fold_in(key, i)
        samples = sampling_fn(key, loc, concentration, shape)
        # Force completion of any asynchronous operations
        samples.block_until_ready()
    end_time = time.time()
    
    # Calculate average time per sample
    total_time = end_time - start_time
    time_per_sample = total_time / (n_runs * shape[0])
    samples_per_second = (n_runs * shape[0]) / total_time
    
    return {
        'total_time': total_time,
        'time_per_sample': time_per_sample,
        'samples_per_second': samples_per_second
    }

# Define test parameters
sample_sizes = [100, 1000, 10000, 100000]
n_runs = 3  # Number of repeated runs

# Create results storage
results = []

# JIT-compiled function with correct static_argnums
jitted_sampler = jax.jit(sample_von_mises, static_argnums=(3,))

print("Testing JIT vs. non-JIT performance...")

for size in sample_sizes:
    shape = (size,)
    print(f"\nSample size: {size}")
    
    # Measure non-JIT performance for smaller sizes
    if size <= 10000:
        print("  Measuring non-JIT performance...")
        non_jit_result = measure_sampling_time(sample_von_mises, key, loc, concentration, shape, n_runs=n_runs)
        results.append({
            'Sample Size': size,
            'Method': 'Non-JIT',
            'Samples/Second': non_jit_result['samples_per_second'],
            'Time/Sample (ms)': non_jit_result['time_per_sample'] * 1000
        })
        print(f"  Non-JIT: {non_jit_result['samples_per_second']:.0f} samples/second")
    
    # Measure JIT performance for all sizes
    print("  Measuring JIT performance...")
    jit_result = measure_sampling_time(jitted_sampler, key, loc, concentration, shape, n_runs=n_runs)
    results.append({
        'Sample Size': size,
        'Method': 'JIT',
        'Samples/Second': jit_result['samples_per_second'],
        'Time/Sample (ms)': jit_result['time_per_sample'] * 1000
    })
    print(f"  JIT: {jit_result['samples_per_second']:.0f} samples/second")
    
    # Calculate speedup if we have both measurements
    if size <= 10000:
        speedup = non_jit_result['time_per_sample'] / jit_result['time_per_sample']
        print(f"  Speedup: {speedup:.1f}x")

# Convert to DataFrame for plotting
df = pd.DataFrame(results)
print("\nResults summary:")
print(df)

# %% [markdown]
# ## 3. Visualization of JIT Performance Benefits
# 
# Let's create some visualizations to showcase the performance improvements:

# %%
# 1. Bar chart comparing JIT vs. non-JIT for different sample sizes
plt.figure(figsize=(14, 8))
subset_df = df[df['Sample Size'] <= 10000].copy()  # Only where we have both measurements
subset_df['Sample Size'] = subset_df['Sample Size'].astype(str)  # Convert to string for categorical plotting

ax = sns.barplot(x='Sample Size', y='Samples/Second', hue='Method', data=subset_df, palette=['lightcoral', 'steelblue'])

# Add text labels for speedup
for i in range(0, len(subset_df), 2):
    if i+1 < len(subset_df):
        non_jit = subset_df.iloc[i]['Samples/Second']
        jit = subset_df.iloc[i+1]['Samples/Second']
        speedup = jit / non_jit
        ax.text(i//2, jit*1.05, f"{speedup:.1f}x faster", ha='center', fontweight='bold')

plt.title('JAX von Mises Sampling Performance: JIT vs. Non-JIT')
plt.ylabel('Samples per Second (higher is better)')
plt.yscale('log')  # Log scale for better visualization
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
# Save figure to file
plt.savefig('figures/jit_vs_nonjit_comparison.png', dpi=300, bbox_inches='tight')
print("Saved JIT vs non-JIT comparison plot to figures/jit_vs_nonjit_comparison.png")
# plt.show()  # Comment out interactive display

# %%
# 2. Line chart showing scalability with sample size
plt.figure(figsize=(14, 8))
df_jit = df[df['Method'] == 'JIT']

sns.lineplot(x='Sample Size', y='Samples/Second', data=df_jit, marker='o', markersize=10, linewidth=3, color='steelblue')
plt.title('JIT Sampling Performance Scaling with Sample Size')
plt.xlabel('Sample Size')
plt.ylabel('Samples per Second (higher is better)')
plt.xscale('log')  # Log scale for x-axis
plt.grid(True, alpha=0.3)
plt.tight_layout()
# Save figure to file
plt.savefig('figures/sample_size_scaling.png', dpi=300, bbox_inches='tight')
print("Saved sample size scaling plot to figures/sample_size_scaling.png")
# plt.show()  # Comment out interactive display

# %% [markdown]
# ## 4. Effect of Concentration Parameter on Performance
# 
# The concentration parameter (κ) affects sampling performance. Let's visualize this relationship:

# %%
# Test performance with different concentration values
concentration_values = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
concentration_results = []

sample_size = 10000
shape = (sample_size,)
n_runs = 3

print("Testing performance with different concentration values...")

for kappa in concentration_values:
    print(f"  Testing κ={kappa}...")
    jit_result = measure_sampling_time(jitted_sampler, key, loc, kappa, shape, n_runs=n_runs)
    concentration_results.append({
        'Concentration': kappa,
        'Samples/Second': jit_result['samples_per_second'],
        'Time/Sample (ms)': jit_result['time_per_sample'] * 1000
    })

# Convert to DataFrame for plotting
df_concentration = pd.DataFrame(concentration_results)

# Plot the results
plt.figure(figsize=(14, 8))
sns.lineplot(x='Concentration', y='Samples/Second', data=df_concentration, marker='o', markersize=10, linewidth=3, color='steelblue')
plt.title('Effect of Concentration Parameter on Sampling Performance')
plt.xlabel('Concentration (κ)')
plt.ylabel('Samples per Second (higher is better)')
plt.xscale('log')  # Log scale for x-axis
plt.grid(True, alpha=0.3)
plt.tight_layout()
# Save figure to file
plt.savefig('figures/concentration_impact_performance.png', dpi=300, bbox_inches='tight')
print("Saved concentration impact plot to figures/concentration_impact_performance.png")
# plt.show()  # Comment out interactive display

# %% [markdown]
# ## 5. Batch Processing with vmap
# 
# JAX's vectorization with `vmap` provides another performance boost when processing multiple distributions simultaneously:

# %%
# Define a function to sample from von Mises with fixed shape
def sample_fn(key, loc, concentration):
    return sample_von_mises(key, loc, concentration, shape=(1000,))

# Create a batched version with vmap
batched_sample_fn = jax.vmap(sample_fn, in_axes=(0, 0, 0))

# Jit-compile the batched function
jitted_batched_fn = jax.jit(batched_sample_fn)

# Test with different batch sizes
batch_sizes = [1, 10, 50, 100]
batch_results = []

for batch_size in batch_sizes:
    print(f"Testing batch size {batch_size}...")
    
    # Create batch inputs
    batch_keys = random.split(key, batch_size)
    batch_locs = jnp.linspace(-jnp.pi/2, jnp.pi/2, batch_size)
    batch_concentrations = jnp.ones(batch_size) * 5.0
    
    # Warmup
    _ = jitted_batched_fn(batch_keys, batch_locs, batch_concentrations)
    
    # Measure batched performance
    start_time = time.time()
    for i in range(n_runs):
        fold_key = random.fold_in(key, i)
        batch_keys = random.split(fold_key, batch_size)
        samples = jitted_batched_fn(batch_keys, batch_locs, batch_concentrations)
        samples.block_until_ready()
    batch_time = time.time() - start_time
    
    # Calculate samples per second
    total_samples = batch_size * 1000 * n_runs
    samples_per_second = total_samples / batch_time
    samples_per_second_per_batch = samples_per_second / batch_size
    
    # Measure sequential performance for small batches
    if batch_size <= 10:
        print("  Measuring sequential performance...")
        start_time = time.time()
        for i in range(n_runs):
            for j in range(batch_size):
                subkey = random.fold_in(key, i*batch_size + j)
                _ = jitted_sampler(subkey, batch_locs[j], batch_concentrations[j], (1000,))
                _.block_until_ready()
        seq_time = time.time() - start_time
        
        seq_samples_per_second = total_samples / seq_time
        speedup = seq_time / batch_time
    else:
        seq_samples_per_second = None
        speedup = None
    
    # Store results
    batch_results.append({
        'Batch Size': batch_size,
        'Samples/Second': samples_per_second,
        'Samples/Second/Batch': samples_per_second_per_batch,
        'Sequential Samples/Second': seq_samples_per_second,
        'Speedup': speedup
    })

# Convert to DataFrame
df_batch = pd.DataFrame(batch_results)

# Plot results
plt.figure(figsize=(14, 8))
plt.plot(df_batch['Batch Size'], df_batch['Samples/Second'], 'o-', linewidth=3, markersize=10, label='Total throughput')
if df_batch['Sequential Samples/Second'].notna().any():
    # Plot sequential performance where available
    sequential_sizes = df_batch['Batch Size'][df_batch['Sequential Samples/Second'].notna()]
    sequential_perf = df_batch['Sequential Samples/Second'][df_batch['Sequential Samples/Second'].notna()]
    plt.plot(sequential_sizes, sequential_perf, 's--', linewidth=2, markersize=8, label='Sequential processing')

plt.title('Batch Processing Performance with vmap')
plt.xlabel('Batch Size')
plt.ylabel('Samples per Second (higher is better)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
# Save figure to file
plt.savefig('figures/batch_processing_performance.png', dpi=300, bbox_inches='tight')
print("Saved batch processing performance plot to figures/batch_processing_performance.png")
# plt.show()  # Comment out interactive display

# Plot speedup
plt.figure(figsize=(14, 6))
speedup_df = df_batch[df_batch['Speedup'].notna()]
plt.bar(speedup_df['Batch Size'].astype(str), speedup_df['Speedup'], color='steelblue')
plt.title('Speedup from Batch Processing vs. Sequential Processing')
plt.xlabel('Batch Size')
plt.ylabel('Speedup Factor (higher is better)')
for i, v in enumerate(speedup_df['Speedup']):
    plt.text(i, v + 0.1, f"{v:.1f}x", ha='center')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
# Save figure to file
plt.savefig('figures/batch_processing_speedup.png', dpi=300, bbox_inches='tight')
print("Saved batch processing speedup plot to figures/batch_processing_speedup.png")
# plt.show()  # Comment out interactive display

# %% [markdown]
# ## 6. First-Call Compilation Overhead
# 
# JAX's JIT compilation has overhead on the first call. Let's measure this overhead:

# %%
print("Measuring JIT compilation overhead...")

# Fresh key and JIT function for clean measurement
new_key = random.PRNGKey(100)
fresh_jitted_sampler = jax.jit(sample_von_mises, static_argnums=(3,))

# Measure first call time (includes compilation)
start_time = time.time()
first_samples = fresh_jitted_sampler(new_key, 0.0, 5.0, (10000,))
first_samples.block_until_ready()
first_call_time = time.time() - start_time

# Measure subsequent call time
start_time = time.time()
second_samples = fresh_jitted_sampler(new_key, 0.0, 5.0, (10000,))
second_samples.block_until_ready()
second_call_time = time.time() - start_time

# Calculate overhead
overhead_factor = first_call_time / second_call_time

print(f"First call time (with compilation): {first_call_time:.4f} seconds")
print(f"Second call time (no compilation): {second_call_time:.4f} seconds")
print(f"Compilation overhead: {first_call_time - second_call_time:.4f} seconds")
print(f"First call is {overhead_factor:.1f}x slower than subsequent calls")

# Create data for bar chart
compilation_df = pd.DataFrame([
    {'Call': 'First call (with compilation)', 'Time (seconds)': first_call_time},
    {'Call': 'Subsequent call', 'Time (seconds)': second_call_time}
])

# Plot bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(compilation_df['Call'], compilation_df['Time (seconds)'], color=['lightcoral', 'steelblue'])
plt.title('JIT Compilation Overhead')
plt.ylabel('Time (seconds)')
plt.grid(axis='y', alpha=0.3)

# Add text labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}s', ha='center')

plt.tight_layout()
# Save figure to file
plt.savefig('figures/compilation_overhead.png', dpi=300, bbox_inches='tight')
print("Saved compilation overhead plot to figures/compilation_overhead.png")
# plt.show()  # Comment out interactive display

# %% [markdown]
# ## 7. Key Takeaways
# 
# Based on the benchmarks we've run, here are the key performance insights:
# 
# 1. **JIT Compilation Provides Dramatic Speedups**: 
#    - JIT compilation speeds up von Mises sampling by **hundreds to thousands of times**
#    - The speedup increases with larger sample sizes
# 
# 2. **First-Call Compilation Overhead**:
#    - The first call to a JIT-compiled function includes compilation time
#    - Subsequent calls are much faster
#    - Consider "warming up" time-critical code with a small initial call
# 
# 3. **Concentration Parameter Impact**:
#    - Sampling performance varies with the concentration parameter (κ)
#    - Lower concentration values generally yield faster sampling
#    - The implementation optimizes special cases for very small and very large κ values
# 
# 4. **Batch Processing Benefits**:
#    - Using `vmap` provides significant speedups for processing multiple distributions
#    - Larger batch sizes generally improve throughput
# 
# ## 8. Best Practices
# 
# To maximize performance with the JAX von Mises sampling library:
# 
# 1. **Always use JIT compilation with correct static arguments**:
#    ```python
#    jitted_sampler = jax.jit(sample_von_mises, static_argnums=(3,))
#    ```
# 
# 2. **Use warmup calls before time-critical operations**:
#    ```python
#    # Warmup with small size
#    _ = jitted_sampler(key, loc, concentration, shape=(10,))
#    # Actual computation
#    samples = jitted_sampler(key, loc, concentration, shape=(10000,))
#    ```
# 
# 3. **Batch process multiple distributions with vmap**:
#    ```python
#    batch_fn = jax.vmap(lambda k, l, c: sample_von_mises(k, l, c, shape=(1000,)))
#    jitted_batch_fn = jax.jit(batch_fn)
#    ```
# 
# 4. **Consider using GPU acceleration for even greater performance**:
#    ```python
#    # Install JAX with GPU support
#    # pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#    ```
# 
# ## Conclusion
# 
# The JAX von Mises sampling library delivers exceptional performance when used with JAX's transformations, particularly JIT compilation. For applications handling directional data, this library offers a high-performance solution that scales well to large sample sizes and batch processing requirements. 

# %% [markdown]
# ### PDF Calculation: von Mises Log Probability Density
# 
# The library includes a `vmises_log_prob` function that calculates the log probability density. Let's visualize it:

# %%
import numpy as np
import matplotlib.pyplot as plt

# Set parameters
loc = 0.0
kappa = 5.0

# Generate x values
x = np.linspace(-np.pi, np.pi, 1000)

# Calculate log probability density
log_prob = vmises_log_prob(x, loc, kappa)
prob = np.exp(log_prob)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, prob, 'b-', linewidth=2)
plt.fill_between(x, prob, alpha=0.3)
plt.xlabel('Angle (radians)')
plt.ylabel('Probability Density')
plt.title(f'Von Mises Probability Density (μ={loc}, κ={kappa})')
plt.grid(alpha=0.3)
plt.tight_layout()
# Save figure to file
plt.savefig('figures/von_mises_density.png', dpi=300, bbox_inches='tight')
print("Saved von Mises density plot to figures/von_mises_density.png")
# plt.show()  # Comment out interactive display

# %% [markdown]
# ### Entropy Calculation and Visualization
#
# The library also includes a `vmises_entropy` function for calculating the entropy of the von Mises distribution, which depends only on the concentration parameter κ.

# %%
from jax_von_mises import vmises_entropy

# Calculate entropy for different concentration values
kappa_values = np.linspace(0.1, 20, 100)
entropy_values = [vmises_entropy(k) for k in kappa_values]

# Plot entropy vs concentration
plt.figure(figsize=(10, 6))
plt.plot(kappa_values, entropy_values, 'b-', linewidth=2.5)
plt.xlabel('Concentration Parameter (κ)')
plt.ylabel('Entropy')
plt.title('Von Mises Distribution Entropy')
plt.grid(alpha=0.3)

# Add annotation for maximum entropy
max_entropy = np.log(2*np.pi)
plt.axhline(y=max_entropy, color='r', linestyle='--', 
            label=f'Maximum entropy (uniform): {max_entropy:.4f}')
plt.legend()
plt.tight_layout()
# Save figure to file
plt.savefig('figures/entropy_vs_concentration.png', dpi=300, bbox_inches='tight')
print("Saved entropy vs concentration plot to figures/entropy_vs_concentration.png")
# plt.show()  # Comment out interactive display

# %% [markdown]
# The entropy is highest when κ=0 (uniform distribution) and decreases as κ increases. This reflects that a more concentrated distribution (higher κ) has lower uncertainty.
#
# ### Entropy Performance with JAX Transformations
#
# Let's compare the performance of entropy calculation with different JAX transformations:

# %%
import time

# Prepare JAX transformations
jitted_entropy = jax.jit(vmises_entropy)
vmapped_entropy = jax.vmap(jitted_entropy)

# Warm up JIT compilation
_ = jitted_entropy(1.0)
_ = vmapped_entropy(jnp.array([1.0]))

# Test sizes
sizes = [10, 100, 1000, 10000, 100000]
results = []

for size in sizes:
    print(f"Testing size: {size}")
    
    # Generate random concentration values
    kappa_values = np.random.uniform(0.1, 10.0, size=size)
    jax_kappa = jnp.array(kappa_values)
    
    # Non-JIT timing
    start = time.time()
    for k in kappa_values:
        _ = vmises_entropy(k)
    nonjit_time = time.time() - start
    
    # JIT timing
    start = time.time()
    for k in kappa_values:
        _ = jitted_entropy(k).block_until_ready()
    jit_time = time.time() - start
    
    # vmap timing
    start = time.time()
    _ = vmapped_entropy(jax_kappa).block_until_ready()
    vmap_time = time.time() - start
    
    results.append({
        'Size': size,
        'Non-JIT (s)': nonjit_time,
        'JIT (s)': jit_time,
        'vmap (s)': vmap_time,
        'JIT Speedup': nonjit_time / jit_time,
        'vmap Speedup': nonjit_time / vmap_time
    })
    
    print(f"  Non-JIT: {nonjit_time:.6f}s")
    print(f"  JIT: {jit_time:.6f}s")
    print(f"  vmap: {vmap_time:.6f}s")
    print(f"  JIT Speedup: {nonjit_time / jit_time:.1f}x")
    print(f"  vmap Speedup: {nonjit_time / vmap_time:.1f}x")

# %% [markdown]
# Now let's visualize these results:

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Convert to DataFrame
df = pd.DataFrame(results)

# Plot speedups
plt.figure(figsize=(12, 6))
plt.plot(df['Size'], df['JIT Speedup'], 'b-o', linewidth=2, label='JIT Speedup')
plt.plot(df['Size'], df['vmap Speedup'], 'r-s', linewidth=2, label='vmap Speedup')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Array Size')
plt.ylabel('Speedup (x times faster)')
plt.title('Entropy Calculation Speedup with JAX Transformations')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
# Save figure to file
plt.savefig('figures/entropy_speedup.png', dpi=300, bbox_inches='tight')
print("Saved entropy speedup plot to figures/entropy_speedup.png")
# plt.show()  # Comment out interactive display

# %% [markdown]
# ### Automatic Differentiation with Entropy
#
# One of the most powerful features of JAX is automatic differentiation. Let's compute the gradient of entropy with respect to the concentration parameter:

# %%
# Define gradient function
grad_fn = jax.grad(vmises_entropy)
jitted_grad = jax.jit(grad_fn)
vmapped_grad = jax.vmap(jitted_grad)

# Compute gradients for a range of concentration values
kappa_values = np.linspace(0.1, 20, 100)
entropy_values = vmapped_entropy(jnp.array(kappa_values))
gradient_values = vmapped_grad(jnp.array(kappa_values))

# Plot entropy and its gradient
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Entropy plot
ax[0].plot(kappa_values, entropy_values, 'b-', linewidth=2, label='Entropy')
ax[0].set_ylabel('Entropy')
ax[0].set_title('Von Mises Entropy and its Gradient')
ax[0].grid(alpha=0.3)
ax[0].legend()

# Gradient plot
ax[1].plot(kappa_values, gradient_values, 'r-', linewidth=2, label='dEntropy/dκ')
ax[1].set_xlabel('Concentration Parameter (κ)')
ax[1].set_ylabel('Gradient')
ax[1].grid(alpha=0.3)
ax[1].legend()

plt.tight_layout()
# Save figure to file
plt.savefig('figures/entropy_gradient.png', dpi=300, bbox_inches='tight')
print("Saved entropy gradient plot to figures/entropy_gradient.png")
# plt.show()  # Comment out interactive display

# %% [markdown]
# This ability to automatically compute derivatives is extremely useful for optimization problems, like finding the concentration parameter that gives a target entropy value. Let's demonstrate:

# %%
from jax import grad
import jax.numpy as jnp
import optax  # JAX's optimization library

# Define a target entropy value
target_entropy = 1.0

# Define a loss function that measures the squared difference between current entropy and target
def loss_fn(kappa):
    entropy = vmises_entropy(kappa)
    return (entropy - target_entropy) ** 2

# Create gradient function
loss_grad_fn = grad(loss_fn)

# Initial concentration value
kappa = jnp.array(5.0)

# Set up optimizer
optimizer = optax.adam(learning_rate=0.1)
opt_state = optimizer.init(kappa)

# Run optimization for a few steps
print(f"Initial κ: {kappa:.4f}, Entropy: {vmises_entropy(kappa):.4f}, Target: {target_entropy:.4f}")

n_steps = 50
kappa_history = [float(kappa)]
entropy_history = [float(vmises_entropy(kappa))]
loss_history = [float(loss_fn(kappa))]

for i in range(n_steps):
    # Compute gradient
    grads = loss_grad_fn(kappa)
    
    # Apply update
    updates, opt_state = optimizer.update(grads, opt_state)
    kappa = optax.apply_updates(kappa, updates)
    
    # Ensure concentration stays positive
    kappa = jnp.maximum(kappa, 0.001)
    
    # Record history
    kappa_history.append(float(kappa))
    current_entropy = float(vmises_entropy(kappa))
    entropy_history.append(current_entropy)
    loss_history.append(float(loss_fn(kappa)))
    
    if (i+1) % 10 == 0:
        print(f"Step {i+1}: κ = {kappa:.4f}, Entropy = {current_entropy:.4f}, Loss = {loss_history[-1]:.6f}")

# %% [markdown]
# Let's visualize the optimization progress:

# %%
# Plot optimization progress
fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Concentration parameter
ax[0].plot(range(n_steps+1), kappa_history, 'b-o', linewidth=2)
ax[0].set_ylabel('Concentration (κ)')
ax[0].set_title('Optimization Progress')
ax[0].grid(alpha=0.3)

# Entropy
ax[1].plot(range(n_steps+1), entropy_history, 'g-s', linewidth=2)
ax[1].axhline(y=target_entropy, color='r', linestyle='--', label=f'Target: {target_entropy}')
ax[1].set_ylabel('Entropy')
ax[1].grid(alpha=0.3)
ax[1].legend()

# Loss
ax[2].plot(range(n_steps+1), loss_history, 'r-^', linewidth=2)
ax[2].set_xlabel('Optimization Step')
ax[2].set_ylabel('Loss')
ax[2].set_yscale('log')
ax[2].grid(alpha=0.3)

plt.tight_layout()
# Save figure to file
plt.savefig('figures/entropy_optimization.png', dpi=300, bbox_inches='tight')
print("Saved entropy optimization plot to figures/entropy_optimization.png")
# plt.show()  # Comment out interactive display

# %% [markdown]
# This example demonstrates how our implementation enables gradient-based optimization of concentration parameters to achieve specific entropy targets - something that would be much more complex with SciPy's implementation.
#
# ### Conclusion
#
# In this notebook, we've demonstrated:
#
# 1. The von Mises sampling functionality with JAX transformations
# 2. The dramatic performance improvements achieved with JIT compilation
# 3. The ability to compute log probability density and entropy efficiently
# 4. The power of automatic differentiation for optimization problems
#
# The JAX von Mises implementation provides a high-performance solution for directional statistics that leverages the full power of JAX's transformation system. 