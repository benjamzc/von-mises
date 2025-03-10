# JAX von Mises Examples

This directory contains example scripts demonstrating how to use the JAX von Mises sampling library.

## Examples Directory

This directory contains various examples demonstrating how to use the von Mises JAX library for directional statistics. These examples cover basic usage, performance benchmarks, and integration with neural networks.

### Performance Showcase

The `performance_showcase.py` script demonstrates the performance characteristics of the von Mises JAX library, including:

1. Basic sampling from the von Mises distribution
2. JIT vs. non-JIT performance benchmarking
3. Visualization of performance scaling
4. Measurement of compilation overhead
5. Comparison with SciPy's vonmises implementation

To run this script:
```bash
python performance_showcase.py
```

To convert it to a Jupyter notebook:
```bash
pip install p2j  # Install Python-to-Jupyter conversion tool
p2j performance_showcase.py
jupyter notebook performance_showcase.ipynb
```

### Entropy Calculation Example

The `entropy_example.py` script demonstrates the entropy calculation functionality for the von Mises distribution:

1. Comparison of our JAX implementation with SciPy's implementation
2. Performance benchmarks between implementations
3. Demonstration of JAX transformations with entropy (JIT, vmap, grad)
4. Example of using entropy in custom loss functions
5. Visualization of entropy values for different concentration parameters

To run this script:
```bash
python entropy_example.py
```

This example showcases how our JAX implementation can be used as a drop-in replacement for SciPy's implementation while enabling additional capabilities like automatic differentiation.

### JIT Compilation Test

The `jit_test.py` script demonstrates proper JIT compilation of the von Mises sampling functions and provides a simple performance comparison:

```bash
python jit_test.py
```

Example output:
```
Non-JIT time: 1.537 seconds
JIT time: 0.0003 seconds
Speedup: 5123.33x
```

### SciPy Comparison

The scripts in the `../figures/` directory include code for comparing our JAX implementation with SciPy's vonmises implementation:

```bash
python ../figures/generate_scipy_comparison.py
```

This will generate:
- Performance comparison charts for sampling and PDF computation
- Feature comparison table
- Distribution visualization comparison

Results are saved in the `../figures/` directory and referenced in the blog post.

## Blog Post and Visualization Assets

The `../figures/` directory contains visualization scripts and images for the blog post:

- `generate_distribution_plot.py`: Creates visualization of the von Mises distribution
- `generate_benchmark_plots.py`: Creates performance benchmark visualizations

The full blog post can be found at `../blog_post.md`, which showcases the dramatic performance improvements achieved with JAX transformations.

## Running Examples with GPU Support

To run these examples with GPU support, first install JAX with the appropriate GPU variant:

```bash
# For NVIDIA GPUs with CUDA
pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For AMD GPUs with ROCm
pip install "jax[rocm]" -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html
```

Or use the provided installation script:

```bash
chmod +x install_gpu.sh
./install_gpu.sh
```

## Troubleshooting

### Common issues:

1. **JIT compilation errors**:
   - Ensure you use `static_argnums=(3,)` when JIT-compiling `sample_von_mises`
   - See the [JIT Compilation Guide](../docs/jit_guide.rst) for details

2. **GPU not detected**:
   - Verify JAX is installed with GPU support (`jax.devices()` should list GPU devices)
   - Check CUDA/ROCm installation
   - Set environment variables if needed:
     ```bash
     export XLA_FLAGS="--xla_gpu_cuda_data_dir=/path/to/cuda"
     ```

3. **Memory errors**:
   - Reduce sample size for large-scale operations
   - Use batched processing for very large sample counts
   
4. **Performance issues**:
   - Remember that first JIT call includes compilation overhead
   - Use warmup calls before time-critical operations
   - Try setting `XLA_FLAGS="--xla_gpu_autotune=true"` 