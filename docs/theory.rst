Von Mises Distribution Theory
===========================

Mathematical Background
----------------------

The von Mises distribution, also known as the circular normal distribution, is a continuous probability distribution on the circle. Named after Richard von Mises, it serves as the circular analogue to the normal distribution and is widely used in directional statistics for modeling angular data.

The von Mises distribution is particularly useful for:

- Modeling directions in two dimensions
- Analyzing circular or periodic data
- Applications in biology, geophysics, meteorology, and computer vision
- Working with bearings, compass directions, seasonal patterns, or phases

Probability Density Function
---------------------------

The probability density function (PDF) of the von Mises distribution is given by:

.. math::

   f(x | \mu, \kappa) = \frac{1}{2\pi I_0(\kappa)}e^{\kappa \cos(x - \mu)}

where:

- :math:`x` is the angle in radians in the range :math:`[-\pi, \pi]`
- :math:`\mu` is the mean direction (location parameter)
- :math:`\kappa` is the concentration parameter (analogous to the inverse of variance)
- :math:`I_0(\kappa)` is the modified Bessel function of the first kind of order 0

The log probability density function, as implemented in our library, is:

.. math::

   \log f(x | \mu, \kappa) = \kappa \cos(x - \mu) - \log(2\pi I_0(\kappa))

Key Properties
------------

Mean Direction
^^^^^^^^^^^^^

The mean direction of the von Mises distribution is simply :math:`\mu`. This parameter determines the location of the peak of the distribution on the circle.

Concentration Parameter
^^^^^^^^^^^^^^^^^^^^^

The concentration parameter :math:`\kappa` determines how concentrated the distribution is around the mean direction:

- When :math:`\kappa = 0`, the distribution is uniform around the circle
- As :math:`\kappa` increases, the distribution becomes more concentrated around the mean
- For large :math:`\kappa` (> 100), the distribution closely approximates a normal distribution with mean :math:`\mu` and variance :math:`1/\kappa`

Circular Variance
^^^^^^^^^^^^^^^

The circular variance is a measure of dispersion for the von Mises distribution, defined as:

.. math::

   V = 1 - \frac{I_1(\kappa)}{I_0(\kappa)}

where :math:`I_1(\kappa)` is the modified Bessel function of the first kind of order 1.

For small values of :math:`\kappa`, the circular variance approaches 1, indicating high dispersion. For large values of :math:`\kappa`, the circular variance approaches :math:`1/\kappa`, similar to a normal distribution.

Entropy
^^^^^^^

The entropy of the von Mises distribution measures the uncertainty or randomness. It is defined as:

.. math::

   H(\kappa) = -\kappa \frac{I_1(\kappa)}{I_0(\kappa)} + \log(2\pi I_0(\kappa))

The entropy depends only on the concentration parameter :math:`\kappa` and not on the mean direction :math:`\mu`. As :math:`\kappa` increases, the entropy decreases, reflecting the increased certainty about the direction.

Key entropy properties:

- When :math:`\kappa \to 0` (uniform distribution), the entropy approaches :math:`\log(2\pi)`, which is the maximum entropy value
- When :math:`\kappa \to \infty` (highly concentrated), the entropy approaches that of a normal distribution with variance :math:`1/\kappa`
- Entropy is useful for information-theoretic analysis and constructing maximum entropy models

Relationship to Other Distributions
---------------------------------

Uniform Distribution
^^^^^^^^^^^^^^^^^^

When :math:`\kappa = 0`, the von Mises distribution reduces to the uniform distribution on the circle.

Normal Distribution
^^^^^^^^^^^^^^^^^

As :math:`\kappa` becomes large, the von Mises distribution approximates a wrapped normal distribution with variance :math:`1/\kappa`. For practical purposes, when :math:`\kappa > 100`, it can be approximated by a normal distribution.

Wrapped Cauchy Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^

The wrapped Cauchy distribution serves as an envelope for rejection sampling from the von Mises distribution. The Best-Fisher algorithm uses this relationship for efficient sampling.

Sampling Algorithm
----------------

Best-Fisher Method
^^^^^^^^^^^^^^^^

Our implementation uses the Best-Fisher algorithm (1979) for sampling from the von Mises distribution. This method employs rejection sampling with a wrapped Cauchy envelope.

The algorithm proceeds as follows:

1. Compute the optimal parameter :math:`p` for the wrapped Cauchy envelope:

   .. math::
   
      p = \frac{\kappa}{2} + \sqrt{1 + \left(\frac{\kappa}{2}\right)^2}

2. Generate a sample :math:`z` from the wrapped Cauchy distribution with parameter :math:`p`:

   a. Generate a uniform random number :math:`u \in [0, 1]`
   b. Compute :math:`z = 2 \arctan\left( \frac{p \tan(\pi u)}{1+\sqrt{1-p^2}} \right)`

3. Accept the sample with probability:

   .. math::
   
      \frac{f_{VM}(z | 0, \kappa)}{Mg_{WC}(z | p)} = \frac{e^{\kappa \cos(z)}}{e^{\kappa \cdot r}}

   where :math:`r = \frac{1 + \sqrt{1-p^2} \cos(z)}{1 + \sqrt{1-p^2}}` and :math:`M` is a normalization constant

4. If accepted, return :math:`z + \mu` (mod :math:`2\pi`) to obtain a sample from von Mises(:math:`\mu`, :math:`\kappa`)
5. If rejected, repeat from step 2

Special Cases
^^^^^^^^^^^

Our implementation includes optimizations for special cases:

- **Very Small Concentration** (:math:`\kappa < 1e-4`): For near-uniform distributions, we directly sample from the uniform distribution on :math:`[-\pi, \pi]`.
- **Very Large Concentration** (:math:`\kappa > 100`): For highly concentrated distributions, we use a normal approximation with mean :math:`\mu` and standard deviation :math:`1/\sqrt{\kappa}`.

Acceptance Rate Analysis
^^^^^^^^^^^^^^^^^^^^^^

The Best-Fisher algorithm has an approximately constant acceptance rate of 65.77% across a wide range of concentration values. This consistent efficiency makes it particularly suitable for sampling-intensive applications like Monte Carlo methods and machine learning.

JAX Implementation Considerations
-------------------------------

Our JAX implementation addresses several key challenges:

Numerical Stability
^^^^^^^^^^^^^^^^^

To ensure numerical stability across a wide range of concentration values:

1. We use the logarithm of the Bessel function (:math:`\log I_0(\kappa)`) instead of the function itself
2. We apply clipping to prevent division by zero and overflow/underflow issues
3. We implement special handling for extreme concentration values

Control Flow in JAX
^^^^^^^^^^^^^^^^^

JAX requires special handling for control flow constructs:

1. We use :code:`lax.while_loop` for implementing the rejection sampling loop
2. We use :code:`lax.cond` for conditional execution
3. All operations are structured in a functional programming style to be compatible with JAX transformations

Shape Handling
^^^^^^^^^^^^

Our implementation carefully handles broadcasting and shape manipulation:

1. We allow batched parameters (location and concentration)
2. We support arbitrary output shapes through the :code:`shape` parameter
3. We ensure proper broadcasting across batch dimensions for neural network integrations

Computational Efficiency
^^^^^^^^^^^^^^^^^^^^^^

To optimize performance in sampling operations:

1. All functions are compatible with JAX's JIT compilation for significant speedups
2. The implementation supports vectorization via :code:`vmap` for batch processing
3. Multi-device execution is supported through :code:`pmap` for parallel sampling

Applications in Machine Learning
------------------------------

Von Mises distributions are particularly useful in machine learning applications involving:

1. **Directional Data**: Modeling wind directions, compass headings, or rotational data
2. **Periodic Features**: Handling time-of-day, day-of-week, or seasonal patterns
3. **Phase Modeling**: Representing phases in signal processing and time series
4. **Circular Regression**: Predicting angular outputs in computer vision and robotics
5. **Uncertainty Modeling**: Capturing uncertainty in directional predictions

Our :code:`von_mises_layer` function enables easy integration with neural networks that predict directions, providing:

- Temperature-controlled sampling during training
- Deterministic mean prediction during inference
- Support for seamless integration with both Flax and Haiku neural network libraries

References
---------

- Best, D. J., & Fisher, N. I. (1979). Efficient Simulation of the von Mises Distribution. *Applied Statistics, 28(2)*, 152-157.
- Mardia, K. V., & Jupp, P. E. (2000). *Directional Statistics*. John Wiley & Sons.
- Jammalamadaka, S. R., & SenGupta, A. (2001). *Topics in Circular Statistics*. World Scientific.
- Fisher, N. I. (1993). *Statistical Analysis of Circular Data*. Cambridge University Press. 