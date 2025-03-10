#!/usr/bin/env python
"""
Example demonstrating the usage of von Mises entropy functions.

This example shows how to use the JAX von Mises entropy function as a drop-in 
replacement for SciPy's implementation, with additional benefits of JAX transformations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.stats

from jax_von_mises import vmises_entropy, vmises_log_prob

# Set up the figure for comparison
plt.figure(figsize=(12, 8))

# Concentration values for comparison
kappa_values = np.linspace(0.1, 20, 100)

# Computing entropy with SciPy
start_time = time.time()
scipy_entropy = np.array([scipy.stats.vonmises.entropy(kappa) for kappa in kappa_values])
scipy_time = time.time() - start_time
print(f"SciPy computation time: {scipy_time:.6f} seconds")

# Computing entropy with JAX (non-JIT)
start_time = time.time()
jax_entropy = np.array([vmises_entropy(kappa) for kappa in kappa_values])
jax_time = time.time() - start_time
print(f"JAX (non-JIT) computation time: {jax_time:.6f} seconds")

# Computing entropy with JAX (JIT)
jitted_entropy = jax.jit(vmises_entropy)
# Warmup call for compilation
_ = jitted_entropy(1.0)
start_time = time.time()
jitted_jax_entropy = np.array([jitted_entropy(kappa).block_until_ready() for kappa in kappa_values])
jit_time = time.time() - start_time
print(f"JAX (JIT) computation time: {jit_time:.6f} seconds")
print(f"JIT speedup over non-JIT: {jax_time/jit_time:.2f}x")

# Computing entropy with JAX (vmap)
vmapped_entropy = jax.vmap(jitted_entropy)
start_time = time.time()
vmapped_jax_entropy = vmapped_entropy(jnp.array(kappa_values)).block_until_ready()
vmap_time = time.time() - start_time
print(f"JAX (vmap+JIT) computation time: {vmap_time:.6f} seconds")
print(f"vmap speedup over JIT: {jit_time/vmap_time:.2f}x")
print(f"Total JAX speedup over SciPy: {scipy_time/vmap_time:.2f}x")

# Plot the entropy values
plt.subplot(2, 1, 1)
plt.plot(kappa_values, scipy_entropy, 'r-', label='SciPy vonmises.entropy')
plt.plot(kappa_values, jax_entropy, 'b--', label='JAX vmises_entropy')
plt.plot(kappa_values, vmapped_jax_entropy, 'g:', label='JAX vmapped entropy')
plt.xlabel('Concentration (κ)')
plt.ylabel('Entropy')
plt.title('Von Mises Entropy Comparison')
plt.legend()
plt.grid(True)

# Plot the differences
plt.subplot(2, 1, 2)
diff_scipy_jax = np.abs(scipy_entropy - jax_entropy)
diff_scipy_vmap = np.abs(scipy_entropy - vmapped_jax_entropy)
plt.semilogy(kappa_values, diff_scipy_jax, 'b-', label='|SciPy - JAX|')
plt.semilogy(kappa_values, diff_scipy_vmap, 'g-', label='|SciPy - JAX vmap|')
plt.xlabel('Concentration (κ)')
plt.ylabel('Absolute Difference (log scale)')
plt.title('Difference Between Implementations')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Demonstration of vmises_entropy as a drop-in replacement
print("\nUsing JAX von Mises as a drop-in replacement for SciPy:")

# Create a function that would normally use SciPy's vonmises.entropy
def analyze_directional_data_scipy(concentrations):
    """Analyze directional data using SciPy's implementation."""
    return {
        'entropy': np.array([scipy.stats.vonmises.entropy(k) for k in concentrations]),
        'mean_entropy': np.mean([scipy.stats.vonmises.entropy(k) for k in concentrations])
    }

# Create the same function using JAX's implementation as a drop-in replacement
def analyze_directional_data_jax(concentrations):
    """Analyze directional data using JAX's implementation."""
    return {
        'entropy': vmapped_entropy(jnp.array(concentrations)),
        'mean_entropy': jnp.mean(vmapped_entropy(jnp.array(concentrations)))
    }

# Sample data for analysis
sample_concentrations = np.array([0.5, 1.0, 2.0, 5.0, 10.0])

# Run analysis with both implementations
scipy_results = analyze_directional_data_scipy(sample_concentrations)
jax_results = analyze_directional_data_jax(sample_concentrations)

# Print results
print("\nSciPy results:")
print(f"Entropy values: {scipy_results['entropy']}")
print(f"Mean entropy: {scipy_results['mean_entropy']}")

print("\nJAX results:")
print(f"Entropy values: {jax_results['entropy']}")
print(f"Mean entropy: {jax_results['mean_entropy']}")

# Save the figure
plt.savefig('entropy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nDemo showing how to use JAX entropy with gradients (not possible with SciPy):")

@jax.jit
def loss_function(concentration, target_entropy):
    """Loss function based on entropy difference."""
    entropy = vmises_entropy(concentration)
    return (entropy - target_entropy) ** 2

# Get gradient function
grad_fn = jax.grad(loss_function)

# Target entropy
target_entropy = 2.0

# Calculate gradient
concentration = 1.0
gradient = grad_fn(concentration, target_entropy)
print(f"Gradient of loss w.r.t. concentration at κ={concentration}: {gradient}")

print("\nConclusion:")
print("The JAX implementation of vmises_entropy is a drop-in replacement for SciPy's")
print("vonmises.entropy with additional benefits of JAX transformations like:")
print("  - JIT compilation for faster execution")
print("  - vmap for vectorized operations")
print("  - Automatic differentiation")
print("  - GPU/TPU acceleration") 