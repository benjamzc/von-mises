Von Mises Distribution Guide
==========================

Theory
------

The von Mises distribution (also known as the circular normal distribution) is a continuous probability distribution on the circle. It is the circular analogue of the normal distribution and is widely used in directional statistics.

Mathematical Definition
^^^^^^^^^^^^^^^^^^^^^^

The probability density function of the von Mises distribution is given by:

.. math::

   f(x | \mu, \kappa) = \frac{e^{\kappa \cos(x - \mu)}}{2\pi I_0(\kappa)}

where:

- :math:`x` is the angle in radians, ranging from :math:`-\pi` to :math:`\pi`
- :math:`\mu` is the mean direction (location parameter)
- :math:`\kappa` is the concentration parameter (analogous to the inverse of variance)
- :math:`I_0(\kappa)` is the modified Bessel function of the first kind of order 0

Properties
^^^^^^^^^

- **Mean Direction**: The mean direction is :math:`\mu`
- **Circular Variance**: :math:`1 - \frac{I_1(\kappa)}{I_0(\kappa)}`
- **Mode**: The distribution is unimodal with its mode at :math:`x = \mu`
- **Symmetry**: The distribution is symmetric about :math:`x = \mu`

As :math:`\kappa` increases, the distribution becomes more concentrated around the mean direction. When :math:`\kappa = 0`, the distribution is uniform on the circle.

Sampling Algorithm
-----------------

This library implements the Best-Fisher algorithm (1979) for sampling from the von Mises distribution. The algorithm uses a rejection sampling approach with a wrapped Cauchy envelope.

Algorithm Steps
^^^^^^^^^^^^^^

1. Calculate parameter :math:`p = \frac{\kappa}{2} + \sqrt{1 + (\frac{\kappa}{2})^2}` for the wrapped Cauchy envelope
2. Generate a random number from the wrapped Cauchy distribution with parameter :math:`p`
3. Accept or reject based on the ratio of the von Mises density to the envelope density
4. If rejected, repeat steps 2-3

The algorithm has an acceptance rate of approximately 65.77% for all :math:`\kappa > 0`, making it efficient even for large concentration values.

Special Cases
^^^^^^^^^^^^

- **High Concentration**: For very large :math:`\kappa` values (e.g., :math:`\kappa > 100`), the distribution closely approximates a normal distribution, and we use a normal approximation for numerical stability.
- **Zero Concentration**: When :math:`\kappa = 0`, the distribution is uniform on the circle, and we simply sample uniformly from :math:`[-\pi, \pi]`.

JAX Compatibility
----------------

This implementation is specifically designed to be compatible with JAX transformations:

- **jit**: All functions can be just-in-time compiled for faster execution
- **vmap**: Vectorized mapping allows efficient batch processing
- **pmap**: Parallel execution across multiple devices (e.g., GPUs or TPUs)
- **grad**: The implementation supports automatic differentiation

Implementation Details
---------------------

The core implementation follows JAX's functional programming paradigm:

1. **Stateless Operations**: All operations are stateless, consistent with JAX's design
2. **PRNG Key Management**: Careful handling of PRNG keys ensures reproducibility
3. **Batched Parameters**: Support for batched parameters enables neural network integration
4. **Device Compatibility**: The code works on both CPU and GPU/TPU

Performance Considerations
-------------------------

Factors affecting performance:

- **JIT Compilation**: First-time JIT compilation may have overhead, but subsequent calls are much faster
- **Batch Size**: Using appropriate batch sizes can significantly improve throughput
- **PRNG Key Management**: Properly splitting keys is essential for correct statistical properties

For optimal performance with large-scale applications:

1. Use `jax.jit` for single-device execution
2. Use `jax.pmap` for multi-device parallel execution
3. Consider vectorizing operations with `jax.vmap` for processing multiple distributions simultaneously

References
----------

Best, D. J., & Fisher, N. I. (1979). Efficient Simulation of the von Mises Distribution. Applied Statistics, 28(2), 152-157.

Mardia, K. V., & Jupp, P. E. (2000). Directional Statistics. John Wiley & Sons. 