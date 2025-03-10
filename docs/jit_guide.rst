JAX JIT Compilation Guide
======================

This guide explains how to effectively use JAX's Just-In-Time (JIT) compilation with the von Mises sampling library, focusing on common pitfalls and best practices.

Understanding JIT Compilation
---------------------------

JAX's JIT compilation transforms Python functions into optimized, compiled code that can run much faster, especially on accelerators like GPUs and TPUs. The `jax.jit` decorator tells JAX to trace the function's execution, build an optimized computational graph, and compile it for the target hardware.

Benefits of JIT compilation with von Mises sampling:

- **Dramatic speedup**: Often 100-1000x faster than non-JIT execution
- **GPU/TPU acceleration**: Enables efficient hardware acceleration
- **Operation fusion**: Optimizes multiple operations together
- **Memory optimization**: More efficient memory usage

Basic Usage Pattern
-----------------

Here's the basic pattern for using JIT compilation with `sample_von_mises`:

.. code-block:: python

    import jax
    from jax import random
    from jax_von_mises import sample_von_mises
    
    # JIT compilation with static_argnums for shape parameter
    jitted_sampler = jax.jit(sample_von_mises, static_argnums=(3,))
    
    # Generate samples
    key = random.PRNGKey(0)
    samples = jitted_sampler(key, 0.0, 2.0, shape=(10000,))

The Critical ``static_argnums`` Parameter
---------------------------------------

When using JAX JIT with `sample_von_mises`, you **must** specify `static_argnums=(3,)` to correctly handle the `shape` parameter:

.. code-block:: python

    # CORRECT: JIT with static_argnums for shape parameter
    jitted_sampler = jax.jit(sample_von_mises, static_argnums=(3,))
    
    # INCORRECT: Will raise an error!
    jitted_sampler = jax.jit(sample_von_mises)  # Missing static_argnums

Why `static_argnums` is Required
------------------------------

The `shape` parameter in JAX functions must be known at compilation time. When JIT-compiling a function:

1. JAX **traces** the function to analyze data dependencies
2. Without `static_argnums`, JAX tries to trace through the `shape` parameter
3. This causes a `TypeError` because `shape` determines array dimensions, which must be statically known

Common Error Message
------------------

If you forget to use `static_argnums`, you'll see an error like:

.. code-block:: text

    TypeError: Shapes must be 1D sequences of concrete values of integer type, got (Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace>,).
    If using `jit`, try using `static_argnums` or applying `jit` to smaller subfunctions.

This error occurs because JAX can't compile a function with dynamic shapes, as the shape determines memory allocation and execution patterns.

Alternative Approach: Using Named Arguments
----------------------------------------

Instead of using positional argument index with `static_argnums`, you can use named arguments with `static_argnames`:

.. code-block:: python

    # Using static_argnames instead of static_argnums
    jitted_sampler = jax.jit(sample_von_mises, static_argnames=('shape', 'max_iters'))
    
    # Both shape and max_iters are now static
    samples = jitted_sampler(key, 0.0, 2.0, shape=(10000,), max_iters=100)

This approach is often clearer since it doesn't rely on remembering argument positions.

JIT with Other Library Functions
-----------------------------

The same principles apply to other functions in the library:

1. **vmises_log_prob**: No static arguments needed:
   
   .. code-block:: python
   
       jitted_log_prob = jax.jit(vmises_log_prob)

2. **compute_p**: No static arguments needed:
   
   .. code-block:: python
   
       jitted_compute_p = jax.jit(compute_p)

3. **von_mises_layer**: Use static arguments for temperature and training:
   
   .. code-block:: python
   
       from jax_von_mises.nn.integration import von_mises_layer
       
       jitted_layer = jax.jit(von_mises_layer, static_argnames=('temperature', 'training'))

Batching with vmap and JIT
------------------------

When using both `vmap` and `jit` together, you need to be careful about argument order:

.. code-block:: python

    # First vmap, then jit
    batched_fn = jax.vmap(lambda k, l, c: sample_von_mises(k, l, c, shape=(1000,)))
    jitted_batched_fn = jax.jit(batched_fn)
    
    # Or jit with static_argnums directly
    jitted_batched_fn = jax.jit(
        jax.vmap(sample_von_mises, in_axes=(0, 0, 0, None)),
        static_argnums=(3,)
    )

Performance Tips
--------------

To maximize JIT performance:

1. **Reuse jitted functions**: The first call includes compilation overhead, subsequent calls are much faster
2. **Use a warmup call**: Make a small call to trigger compilation before time-critical operations
3. **Fix batch sizes**: If possible, use consistent batch sizes to avoid recompilation
4. **Use appropriate shapes**: Overly large or small shapes may not perform optimally

Example with Warmup
-----------------

Here's an example showing proper JIT use with warmup:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from jax import random
    from jax_von_mises import sample_von_mises
    import time
    
    # JIT compilation with static_argnums
    jitted_sampler = jax.jit(sample_von_mises, static_argnums=(3,))
    
    # Warmup call with small size (triggers compilation)
    key = random.PRNGKey(0)
    _ = jitted_sampler(key, 0.0, 2.0, shape=(10,))
    
    # Actual large-scale sampling (fast, no compilation overhead)
    start = time.time()
    samples = jitted_sampler(key, 0.0, 2.0, shape=(1000000,))
    elapsed = time.time() - start
    print(f"Generated 1M samples in {elapsed:.4f} seconds")
    print(f"Samples per second: {1000000/elapsed:.0f}")

Common JIT Pitfalls
-----------------

1. **Forgetting static_argnums**: Always use `static_argnums=(3,)` with `sample_von_mises`
2. **Recompilation**: Changing shape causes recompilation, so try to use consistent shapes
3. **Compilation delay**: The first call includes compilation time, which can be significant
4. **Memory usage**: JIT can increase peak memory usage during compilation
5. **Debugging difficulty**: Debugging JIT-compiled code is harder

Debugging JIT Issues
------------------

If you encounter JIT-related problems:

1. Try without JIT first to ensure your code works correctly
2. Check all static arguments are correctly specified
3. Add print statements before JIT to verify inputs
4. Use `jax.disable_jit()` context to temporarily disable JIT:

   .. code-block:: python
   
       with jax.disable_jit():
           samples = sample_von_mises(key, 0.0, 2.0, shape=(1000,))

5. For complex cases, break down into smaller functions and JIT each separately

Advanced: JIT and RNG Key Management
---------------------------------

When using JIT with random sampling, properly manage RNG keys:

.. code-block:: python

    # Generate multiple sets of samples with different RNG paths
    def generate_multiple_samples(key, n_sets):
        keys = random.split(key, n_sets)
        
        @jax.jit
        def sample_one(key):
            return sample_von_mises(key, 0.0, 2.0, shape=(1000,))
        
        return jax.vmap(sample_one)(keys)

Conclusion
--------

Proper use of JIT compilation can dramatically accelerate your von Mises sampling operations. The key takeaways are:

1. **Always use `static_argnums=(3,)` with `sample_von_mises`**
2. Consider warmup calls for time-critical code
3. Reuse jitted functions whenever possible
4. Be aware of compilation overhead on the first call

Following these guidelines will help you avoid common errors and achieve maximum performance with the JAX von Mises sampling library. 