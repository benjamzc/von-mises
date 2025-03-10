# Neural Network Integration Example

This document demonstrates how to integrate the von Mises sampling with neural network outputs.

## Building a Simple Neural Network

Let's create a simple neural network that predicts parameters for a von Mises distribution. The network will take 2D inputs and predict the mean direction (μ) and concentration (κ) for the von Mises distribution.

```python
import jax
import jax.numpy as jnp
from jax import random, grad, value_and_grad
import numpy as np
import matplotlib.pyplot as plt

from jax_von_mises import sample_von_mises, vmises_log_prob
from jax_von_mises.nn.integration import von_mises_layer, pmap_compatible_von_mises_sampling

def init_network_params(key, layer_sizes):
    """Initialize parameters for a fully-connected neural network."""
    keys = random.split(key, len(layer_sizes))
    params = []
    
    for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        w_key, b_key = random.split(keys[i])
        scale = 1.0 / np.sqrt(in_size) # Xavier initialization
        params.append({
            'weights': scale * random.normal(w_key, (in_size, out_size)),
            'bias': jnp.zeros(out_size),
        })
    
    return params

def apply_network(params, inputs):
    """Apply a neural network to inputs."""
    x = inputs
    # Apply all layers except the last one with ReLU activation
    for layer_params in params[:-1]:
        x = jnp.dot(x, layer_params['weights']) + layer_params['bias']
        x = jax.nn.relu(x)
    
    # Apply last layer without activation (for loc) and softplus (for concentration)
    final_params = params[-1]
    x = jnp.dot(x, final_params['weights']) + final_params['bias']
    
    # Split outputs into mean and concentration
    mean_logits, conc_logits = jnp.split(x, 2, axis=-1)
    
    # Keep mean unbounded and ensure concentration is positive
    concentration = jax.nn.softplus(conc_logits) + 0.01
    
    return mean_logits, concentration
```

## Generating Synthetic Data

Let's create a synthetic dataset where the angle is related to the 2D input positions.

```python
def generate_synthetic_data(key, n_samples=1000):
    """Generate synthetic data with angular targets."""
    key, subkey = random.split(key)
    
    # Generate 2D inputs in a grid
    x = jnp.linspace(-3, 3, int(jnp.sqrt(n_samples)))
    y = jnp.linspace(-3, 3, int(jnp.sqrt(n_samples)))
    X, Y = jnp.meshgrid(x, y)
    inputs = jnp.column_stack([X.ravel(), Y.ravel()])
    
    # Calculate angles from the origin
    angles = jnp.arctan2(inputs[:, 1], inputs[:, 0])
    
    # Add noise from von Mises distribution
    noise_conc = 10.0  # High concentration = low noise
    angles = sample_von_mises(subkey, angles, noise_conc)
    
    return inputs, angles

# Generate data
key = random.PRNGKey(42)
inputs, angles = generate_synthetic_data(key, n_samples=1000)
```

## Training the Neural Network

We'll train the neural network to predict the von Mises distribution parameters for our synthetic data.
The loss function will be the negative log-likelihood of the von Mises distribution.

```python
def loss_fn(params, inputs, targets):
    """Compute negative log-likelihood loss for von Mises predictions."""
    mean_logits, concentration = apply_network(params, inputs)
    
    # Normalize mean to [-π, π]
    mean = jnp.arctan2(jnp.sin(mean_logits), jnp.cos(mean_logits))
    
    # Compute log probability density
    log_probs = vmises_log_prob(targets, mean, concentration)
    
    # Return negative log-likelihood
    return -jnp.mean(log_probs)

def update(params, inputs, targets, learning_rate=0.01):
    """Update network parameters using gradient descent."""
    grads = grad(loss_fn)(params, inputs, targets)
    
    # Update parameters
    new_params = []
    for layer_params, layer_grads in zip(params, grads):
        new_layer = {}
        for param_name, param_value in layer_params.items():
            new_layer[param_name] = param_value - learning_rate * layer_grads[param_name]
        new_params.append(new_layer)
    
    return new_params

# Initialize network
layer_sizes = [2, 32, 32, 2]  # Last layer outputs 2 values: mean and concentration
key, subkey = random.split(key)
params = init_network_params(subkey, layer_sizes)

# Train the network (in a real implementation, you would run multiple epochs)
batch_inputs = inputs[:100]  # Use a small batch for example
batch_angles = angles[:100]
params = update(params, batch_inputs, batch_angles)
```

## Using the von_mises_layer for Sampling

Now let's use our `von_mises_layer` to sample from the predicted distributions.

```python
# Get predictions for inputs
mean_logits, concentration = apply_network(params, inputs)

# Sample from predicted distributions
key, subkey = random.split(key)
samples, mean = von_mises_layer(subkey, mean_logits, concentration, temperature=1.0, training=True)

# In inference mode (no sampling)
inference_samples, inference_mean = von_mises_layer(
    subkey, mean_logits, concentration, training=False
)
# In inference mode, samples should equal mean
assert jnp.allclose(inference_samples, inference_mean)
```

## Using the pmap-Compatible Function

Finally, let's demonstrate how to use the `pmap_compatible_von_mises_sampling` function for parallel execution.

```python
# Define a wrapper for our network
def model_fn(params, inputs):
    return apply_network(params, inputs)

# For a real pmap example, you would use:
# parallel_fn = jax.pmap(
#     lambda inputs, key: pmap_compatible_von_mises_sampling(
#         model_fn, params, inputs, key, temperature=1.0, training=True
#     )
# )

# For demonstration, we'll use vmap instead
vmap_sample_fn = jax.vmap(
    lambda inputs, key: pmap_compatible_von_mises_sampling(
        model_fn, params, inputs, key, temperature=1.0, training=True
    ),
    in_axes=(0, 0)
)

# Split inputs and keys for batched processing
num_batches = 4
batch_size = len(inputs) // num_batches
input_batches = inputs.reshape(num_batches, batch_size, -1)
keys = random.split(key, num_batches)

# Run batched sampling
batched_samples = vmap_sample_fn(input_batches, keys)
print(f"Batched samples shape: {batched_samples.shape}")
```

## Conclusion

We've demonstrated how to:

1. Train a neural network to predict von Mises distribution parameters
2. Use the `von_mises_layer` to sample from these distributions
3. Use the `pmap_compatible_von_mises_sampling` function for batched processing

This approach can be extended to more complex models and applications where directional data is involved. 