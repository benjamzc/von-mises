#!/usr/bin/env python
"""
Script to generate visualizations for von Mises entropy function.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import os
import time
import jax
import jax.numpy as jnp

from jax_von_mises import vmises_entropy

# Set plotting style
plt.style.use('ggplot')
sns.set_context("notebook", font_scale=1.5)

# Create directory for figures
os.makedirs('figures', exist_ok=True)

def generate_entropy_plot():
    """Generate a plot showing entropy for different concentration values."""
    plt.figure(figsize=(12, 8))
    
    # Concentration values
    kappa_values = np.linspace(0.01, 20, 1000)
    
    # Compute entropy using JAX
    jitted_entropy = jax.jit(vmises_entropy)
    vmapped_entropy = jax.vmap(jitted_entropy)
    entropy_values = vmapped_entropy(jnp.array(kappa_values))
    
    # Plot entropy vs concentration
    plt.plot(kappa_values, entropy_values, 'b-', linewidth=2.5)
    
    # Add key values
    plt.axhline(y=np.log(2*np.pi), color='r', linestyle='--', 
                label=f'Maximum entropy (uniform): {np.log(2*np.pi):.4f}')
    
    # Annotations for specific points
    plt.annotate(f'κ=0: Maximum entropy', 
                 xy=(0.01, np.log(2*np.pi)), 
                 xytext=(4, np.log(2*np.pi)+0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.annotate(f'κ=5: {jitted_entropy(5.0):.4f}', 
                 xy=(5, jitted_entropy(5.0)), 
                 xytext=(7, jitted_entropy(5.0)+0.3),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.xlabel('Concentration Parameter (κ)')
    plt.ylabel('Entropy')
    plt.title('Von Mises Distribution Entropy')
    plt.grid(True)
    plt.ylim(0, np.log(2*np.pi) + 0.5)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('figures/von_mises_entropy.png', dpi=300, bbox_inches='tight')
    print("Saved von_mises_entropy.png")

def generate_entropy_performance_comparison():
    """Compare performance between JAX and SciPy entropy calculations."""
    plt.figure(figsize=(12, 6))
    
    # Test with different array sizes
    array_sizes = [10, 100, 1000, 10000, 100000]
    
    # Prepare results collections
    scipy_times = []
    jax_times = []
    jit_times = []
    vmap_times = []
    
    print("Benchmarking entropy calculation performance...")
    
    # Prepare jitted functions
    jitted_entropy = jax.jit(vmises_entropy)
    # Warmup for JIT compilation
    _ = jitted_entropy(1.0)
    
    for size in array_sizes:
        print(f"Testing array size: {size}")
        
        # Generate random concentration values
        kappa_values = np.random.uniform(0.1, 10.0, size=size)
        jax_kappa = jnp.array(kappa_values)
        
        # SciPy timing
        start_time = time.time()
        _ = [scipy.stats.vonmises.entropy(k) for k in kappa_values]
        scipy_time = time.time() - start_time
        scipy_times.append(scipy_time)
        
        # JAX non-JIT timing
        start_time = time.time()
        _ = [vmises_entropy(k) for k in kappa_values]
        jax_time = time.time() - start_time
        jax_times.append(jax_time)
        
        # JAX JIT timing
        start_time = time.time()
        for k in kappa_values:
            _ = jitted_entropy(k).block_until_ready()
        jit_time = time.time() - start_time
        jit_times.append(jit_time)
        
        # JAX vmap timing
        vmapped_entropy = jax.vmap(jitted_entropy)
        start_time = time.time()
        _ = vmapped_entropy(jax_kappa).block_until_ready()
        vmap_time = time.time() - start_time
        vmap_times.append(vmap_time)
        
        print(f"  SciPy: {scipy_time:.6f}s, JAX: {jax_time:.6f}s, JIT: {jit_time:.6f}s, vmap: {vmap_time:.6f}s")
    
    # Convert to samples per second
    scipy_rate = [size/time for size, time in zip(array_sizes, scipy_times)]
    jax_rate = [size/time for size, time in zip(array_sizes, jax_times)]
    jit_rate = [size/time for size, time in zip(array_sizes, jit_times)]
    vmap_rate = [size/time for size, time in zip(array_sizes, vmap_times)]
    
    # Plot results
    plt.loglog(array_sizes, scipy_rate, 'r-', marker='o', linewidth=2, label='SciPy')
    plt.loglog(array_sizes, jax_rate, 'b--', marker='s', linewidth=2, label='JAX (non-JIT)')
    plt.loglog(array_sizes, jit_rate, 'g-.', marker='^', linewidth=2, label='JAX (JIT)')
    plt.loglog(array_sizes, vmap_rate, 'm:', marker='D', linewidth=2, label='JAX (vmap+JIT)')
    
    plt.xlabel('Array Size')
    plt.ylabel('Calculations per Second (log scale)')
    plt.title('Entropy Calculation Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('figures/entropy_performance.png', dpi=300, bbox_inches='tight')
    print("Saved entropy_performance.png")

def generate_entropy_gradient_demo():
    """Demonstrate automatic differentiation of entropy function."""
    plt.figure(figsize=(10, 6))
    
    # Concentration values
    kappa_values = np.linspace(0.01, 20, 1000)
    
    # Compute entropy 
    jitted_entropy = jax.jit(vmises_entropy)
    vmapped_entropy = jax.vmap(jitted_entropy)
    entropy_values = vmapped_entropy(jnp.array(kappa_values))
    
    # Compute gradient of entropy w.r.t. concentration
    grad_fn = jax.grad(vmises_entropy)
    jitted_grad = jax.jit(grad_fn)
    vmapped_grad = jax.vmap(jitted_grad)
    gradient_values = vmapped_grad(jnp.array(kappa_values))
    
    # Plot entropy and its gradient
    plt.subplot(2, 1, 1)
    plt.plot(kappa_values, entropy_values, 'b-', linewidth=2.5, label='Entropy')
    plt.ylabel('Entropy')
    plt.title('Von Mises Entropy and its Gradient')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(kappa_values, gradient_values, 'r-', linewidth=2.5, label='dEntropy/dκ')
    plt.xlabel('Concentration Parameter (κ)')
    plt.ylabel('Gradient')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('figures/entropy_gradient.png', dpi=300, bbox_inches='tight')
    print("Saved entropy_gradient.png")

if __name__ == "__main__":
    print("Generating von Mises entropy visualizations...")
    generate_entropy_plot()
    generate_entropy_performance_comparison()
    generate_entropy_gradient_demo()
    print("All entropy visualizations complete!") 