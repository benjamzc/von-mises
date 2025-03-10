# JAX von Mises Visualizations

This directory contains scripts and output images for visualizing the JAX von Mises distribution library and its performance.

## Visualization Scripts

### Distribution Visualizations

- **`generate_distribution_plot.py`**: Creates visualizations of the von Mises distribution with different concentration parameters.
  - Output: `von_mises_distribution.png`

- **`generate_entropy_plot.py`**: Creates visualizations related to the entropy of the von Mises distribution.
  - Outputs:
    - `von_mises_entropy.png`: Entropy as a function of concentration parameter
    - `entropy_gradient.png`: Entropy and its gradient with respect to concentration
    - `entropy_performance.png`: Performance comparison of different entropy calculation methods

### Performance Benchmarks

- **`generate_benchmark_plots.py`**: Generates performance benchmark visualizations for the JAX von Mises library.
  - Outputs:
    - `jit_vs_nonjit_bar.png`: JIT vs. non-JIT performance comparison
    - `concentration_impact.png`: Impact of concentration parameter on performance
    - `batch_processing.png`: Performance scaling with batch size
    - `compilation_overhead.png`: Analysis of JIT compilation overhead

### Comparison with SciPy

- **`generate_scipy_comparison.py`**: Compares performance and features between the JAX von Mises implementation and SciPy's `vonmises`.
  - Outputs:
    - `jax_vs_scipy_sampling.png`: Sampling performance comparison
    - `jax_vs_scipy_pdf.png`: PDF computation performance comparison
    - `entropy_comparison.png`: Entropy calculation performance comparison
    - `feature_comparison.png`: Feature comparison table
    - `visualization_comparison.png`: Side-by-side distribution visualization
    - `distribution_overlay.png`: Overlay of JAX and SciPy distributions

## Entropy Visualization Details

The entropy visualizations showcased in this directory demonstrate several key aspects of the von Mises entropy function:

1. **Entropy vs. Concentration**: Shows how entropy decreases as concentration increases, with maximum entropy (log(2π)) at κ=0 (uniform distribution).

2. **Gradient Visualization**: Demonstrates the gradient of entropy with respect to the concentration parameter, which is crucial for gradient-based optimization.

3. **Performance Comparison**: Shows the dramatic performance improvements achieved with JAX transformations:
   - For small arrays (<1000 elements), SciPy's implementation is faster due to lower overhead
   - For moderate arrays (~1000 elements), performance is approximately equal
   - For large arrays (>10,000 elements), JAX with vmap+JIT shows 10-100x speedup

## Using the Visualization Scripts

To generate all visualizations:

```bash
# Generate distribution visualizations
python generate_distribution_plot.py

# Generate entropy visualizations
python generate_entropy_plot.py

# Generate benchmark plots
python generate_benchmark_plots.py

# Generate comparison with SciPy
python generate_scipy_comparison.py
```

These visualizations are also referenced in the blog post (`../blog_post.md`) to illustrate the benefits of the JAX von Mises implementation. 