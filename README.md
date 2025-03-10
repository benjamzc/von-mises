# JAX Von Mises Sampling

A JAX-compatible implementation of von Mises distribution sampling using the Best-Fisher (1979) algorithm. This package is designed specifically for integration with neural networks and parallel execution on GPUs via `jax.pmap`.

## Installation

```bash
pip install jax-von-mises
```

## Features

* Efficient rejection sampling algorithm with ~65.77% acceptance rate
* Fully compatible with JAX transformations (jit, vmap, pmap)
* Optimized for neural network integration
* Well-tested and numerically stable
* Supports both NVIDIA GPUs (CUDA) and AMD GPUs (ROCm)

## Quick Example

```python
import jax
import jax.numpy as jnp
from jax import random
from jax_von_mises import sample_von_mises

# Generate samples
key = random.PRNGKey(0)
loc = 0.0  # mean direction
concentration = 5.0  # concentration parameter
samples = sample_von_mises(key, loc, concentration, shape=(1000,))

# Plot the distribution
import matplotlib.pyplot as plt
plt.hist(samples, bins=50)
plt.title('Von Mises Samples (μ=0, κ=5)')
plt.xlabel('Angle (radians)')
plt.show()
```

## Neural Network Integration

```python
from jax_von_mises.nn.integration import pmap_compatible_von_mises_sampling

def model_fn(params, inputs):
    # Your neural network that outputs (mean, concentration)
    # ... model code ...
    return mean_logits, concentration

# Use with pmap for parallel execution on multiple GPUs
parallel_fn = jax.pmap(
    lambda inputs, key: pmap_compatible_von_mises_sampling(
        model_fn, params, inputs, key
    )
)

# Split inputs and keys across devices
device_inputs = jnp.array_split(inputs, jax.device_count())
device_keys = random.split(key, jax.device_count())

# Run in parallel
samples = parallel_fn(device_inputs, device_keys)
```

## Development Environment

### Development Containers

We provide development container configurations for both NVIDIA (CUDA) and AMD (ROCm) GPUs:

- **Default (CPU)**: Basic configuration for CPU-only development
- **NVIDIA GPU**: Configuration for systems with NVIDIA GPUs using CUDA
- **AMD GPU**: Configuration for systems with AMD GPUs using ROCm

To use these configurations with VS Code:

1. Open this repository in Visual Studio Code
2. Install the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension
3. Open the Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
4. Select `Remote-Containers: Rebuild and Reopen in Container`
5. Choose your preferred configuration variant

See [.devcontainer/README.md](.devcontainer/README.md) for detailed instructions.

## Performance & Benchmarks

The library has been carefully optimized for both correctness and speed. The use of JAX transformations provides significant performance improvements:

- **JIT compilation**: 5,000-7,000x speedup for sampling operations
- **Vectorization**: Efficient batch processing with `vmap`
- **GPU/TPU support**: Massive parallelism for large-scale operations

See the [blog post](blog_post.md) for detailed performance benchmarks and comparison with SciPy's vonmises implementation.

To run the benchmarks yourself:
```bash
# JIT vs non-JIT performance test
python examples/jit_test.py

# Full performance showcase
python examples/performance_showcase.py
```

## Documentation & Examples

For full documentation and examples, visit [https://jax-von-mises.readthedocs.io](https://jax-von-mises.readthedocs.io).

## Contribute

We welcome contributions! Please check out our [GitHub repository](https://github.com/engelberger/von-mises).

## Citation

If you use this package in your research, please cite the original algorithm:

```
Best, D. J., & Fisher, N. I. (1979). Efficient Simulation of the von Mises Distribution. 
Applied Statistics, 28(2), 152-157.
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 