# Parallel Sampling with JAX

This document demonstrates how to efficiently sample from von Mises distributions in parallel using JAX's transformation functions, particularly focusing on `jax.pmap` for multi-device parallelism.

## Basic JAX Transformations

JAX provides several transformations that allow efficient execution of code:

1. `jit` - Just-In-Time compilation for accelerated execution
2. `vmap` - Vectorized mapping for batch processing
3. `pmap` - Parallel mapping across multiple devices (GPUs/TPUs)

Let's see how to use these with von Mises sampling:

```python
import jax
import jax.numpy as jnp
from jax import random, jit, vmap, pmap
import time
import matplotlib.pyplot as plt

from jax_von_mises import sample_von_mises

# Basic sampling
key = random.PRNGKey(42)
loc = 0.0
concentration = 5.0
```

## JIT Compilation

```python
# Define a function to sample
def sample_fn(key, loc, concentration, n_samples):
    return sample_von_mises(key, loc, concentration, shape=(n_samples,))

# Create JIT-compiled version
jitted_sample_fn = jit(sample_fn)

# Compare performance
n_samples = 10000

# Warmup
_ = sample_fn(key, loc, concentration, n_samples)
_ = jitted_sample_fn(key, loc, concentration, n_samples)

# Time regular function
start = time.time()
samples = sample_fn(key, loc, concentration, n_samples)
regular_time = time.time() - start

# Time JIT-compiled function
start = time.time()
jitted_samples = jitted_sample_fn(key, loc, concentration, n_samples)
jit_time = time.time() - start

print(f"Regular: {regular_time:.6f} seconds")
print(f"JIT: {jit_time:.6f} seconds")
print(f"Speedup: {regular_time/jit_time:.1f}x")
```

## Vectorization with vmap

```python
# Define function to sample from multiple distributions
def multi_sample_fn(key, locs, concentrations, n_samples):
    keys = random.split(key, len(locs))
    return vmap(sample_fn, in_axes=(0, 0, 0, None))(keys, locs, concentrations, n_samples)

# Sample from multiple distributions
locs = jnp.array([0.0, jnp.pi/4, jnp.pi/2, 3*jnp.pi/4, jnp.pi])
concentrations = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
n_samples_per_dist = 1000

samples = multi_sample_fn(key, locs, concentrations, n_samples_per_dist)
print(f"Batched samples shape: {samples.shape}")  # (5, 1000)

# Compare with manual loop
def manual_multi_sample(key, locs, concentrations, n_samples):
    keys = random.split(key, len(locs))
    results = []
    for i in range(len(locs)):
        result = sample_fn(keys[i], locs[i], concentrations[i], n_samples)
        results.append(result)
    return jnp.stack(results)

# Warmup
_ = multi_sample_fn(key, locs, concentrations, n_samples_per_dist)
_ = manual_multi_sample(key, locs, concentrations, n_samples_per_dist)

# Time vectorized version
start = time.time()
vectorized_samples = multi_sample_fn(key, locs, concentrations, n_samples_per_dist)
vectorized_time = time.time() - start

# Time manual loop
start = time.time()
manual_samples = manual_multi_sample(key, locs, concentrations, n_samples_per_dist)
manual_time = time.time() - start

print(f"Manual loop: {manual_time:.6f} seconds")
print(f"Vectorized: {vectorized_time:.6f} seconds")
print(f"Speedup: {manual_time/vectorized_time:.1f}x")
```

## Multi-Device Parallelism with pmap

```python
# Get number of available devices
n_devices = jax.device_count()
print(f"Number of devices: {n_devices}")

if n_devices > 1:
    # Define pmap-compatible function
    def parallel_sample_fn(key, loc, concentration, n_samples):
        return sample_von_mises(key, loc, concentration, shape=(n_samples,))
    
    pmapped_sample_fn = pmap(parallel_sample_fn)
    
    # Prepare data for each device
    keys = random.split(key, n_devices)
    locs = jnp.array([0.0, jnp.pi/4, jnp.pi/2, 3*jnp.pi/4][:n_devices])
    concentrations = jnp.array([1.0, 2.0, 5.0, 10.0][:n_devices])
    
    # Run in parallel across devices
    parallel_samples = pmapped_sample_fn(keys, locs, concentrations, n_samples_per_dist)
    print(f"Parallel samples shape: {parallel_samples.shape}")  # (n_devices, n_samples_per_dist)
else:
    print("Multiple devices not available. Using vmap as example.")
    # Use vmap as an illustrative example
    pmapped_sample_fn = vmap(sample_fn, in_axes=(0, 0, 0, None))
    
    # Prepare data
    keys = random.split(key, 4)
    locs = jnp.array([0.0, jnp.pi/4, jnp.pi/2, 3*jnp.pi/4])
    concentrations = jnp.array([1.0, 2.0, 5.0, 10.0])
    
    # Run vectorized version
    parallel_samples = pmapped_sample_fn(keys, locs, concentrations, n_samples_per_dist)
    print(f"Simulated parallel samples shape: {parallel_samples.shape}")
```

## Scaling to Large Batches

For very large batches, we can combine `pmap` and `vmap`:

```python
# Define a function that uses both pmap and vmap
def pmap_vmap_sample(key, locs_array, concentrations_array, n_samples):
    # Split key for each device
    keys = random.split(key, n_devices)
    
    # Run on multiple devices with pmap
    # Each device handles a batch of samples with vmap
    # in_axes=(0, 0, 0, None) means vectorize over the first three arguments
    return pmap(
        lambda key, locs, concs: vmap(
            sample_fn, in_axes=(0, 0, 0, None)
        )(
            random.split(key, locs.shape[0]), locs, concs, n_samples
        )
    )(keys, locs_array, concentrations_array)

if n_devices > 1:
    # Create larger batches per device
    n_per_device = 10
    
    # Reshape data for pmap + vmap
    device_keys = random.split(key, n_devices)
    device_locs = jnp.ones((n_devices, n_per_device)) * jnp.arange(n_devices)[:, None] * jnp.pi/4
    device_concentrations = jnp.ones((n_devices, n_per_device)) * (jnp.arange(n_per_device)[None, :] + 1.0)
    
    # Run with pmap + vmap
    large_batch_samples = pmap_vmap_sample(key, device_locs, device_concentrations, n_samples_per_dist)
    print(f"Large batch samples shape: {large_batch_samples.shape}")  
    # Should be (n_devices, n_per_device, n_samples_per_dist)
else:
    print("Multiple devices not available for pmap + vmap demonstration.")
```

## Neural Network Integration for Parallel Processing

In a real application, you'd often be using neural networks to generate von Mises parameters:

```python
from jax_von_mises.nn.integration import pmap_compatible_von_mises_sampling

# Define a simple model that outputs von Mises parameters
def simple_model(params, inputs):
    # In a real model, this would be a neural network forward pass
    # Here we just use a simple linear transformation
    mean_logits = inputs * params['mean_scale'] + params['mean_bias']
    concentration = jnp.exp(inputs * params['conc_scale'] + params['conc_bias'])
    return mean_logits, concentration

# Initialize model parameters
model_params = {
    'mean_scale': jnp.array([0.5, -0.5]),
    'mean_bias': jnp.array(0.0),
    'conc_scale': jnp.array([0.3, 0.3]),
    'conc_bias': jnp.array(0.5)
}

# Define inputs for each device
inputs_per_device = 8
input_dim = 2

# Create inputs
device_inputs = jnp.ones((n_devices, inputs_per_device, input_dim))
device_keys = random.split(key, n_devices)

if n_devices > 1:
    # Use pmap for parallel execution
    parallel_fn = pmap(
        lambda inputs, key: pmap_compatible_von_mises_sampling(
            simple_model, model_params, inputs, key, temperature=1.0
        )
    )
    
    # Run in parallel
    nn_samples = parallel_fn(device_inputs, device_keys)
    print(f"Neural network parallel samples shape: {nn_samples.shape}")
else:
    print("Multiple devices not available for neural network parallel demonstration.")
    # Simulate with vmap
    vmap_fn = vmap(
        lambda inputs, key: pmap_compatible_von_mises_sampling(
            simple_model, model_params, inputs, key, temperature=1.0
        )
    )
    nn_samples = vmap_fn(device_inputs[0], random.split(key, inputs_per_device))
    print(f"Simulated neural network parallel samples shape: {nn_samples.shape}")
```

## Conclusion

This example has demonstrated how to:

1. Use JAX's transformations (`jit`, `vmap`, `pmap`) with von Mises sampling
2. Achieve significant speedups through vectorization and parallelization
3. Process large batches of samples across multiple devices
4. Integrate with neural networks for parallel sampling

For real applications, you'll want to profile and optimize the batch sizes and distribution of work across devices based on your specific hardware configuration. 