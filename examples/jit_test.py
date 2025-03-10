#!/usr/bin/env python
"""
Test script to verify proper JIT compilation with von Mises sampling.

This script demonstrates the correct way to use JAX's JIT compilation
with the sample_von_mises function, focusing on proper handling of the
shape parameter using static_argnums.

Usage:
    python jit_test.py [--sample-size SIZE] [--iterations ITERS]
"""

import argparse
import time
import jax
import jax.numpy as jnp
from jax import random

try:
    from jax_von_mises import sample_von_mises
except ImportError:
    print("Error: jax_von_mises package not found.")
    print("Please install it with: pip install jax-von-mises")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Test JAX JIT compilation with von Mises sampling")
    parser.add_argument("--sample-size", type=int, default=10000,
                        help="Number of samples (default: 10000)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of sampling iterations (default: 10)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Sample size: {args.sample_size}")
    print(f"Iterations: {args.iterations}")
    print("\n" + "=" * 50)
    
    # Create PRNG key
    key = random.PRNGKey(42)
    
    # Test parameters
    loc = 0.0
    concentration = 5.0
    shape = (args.sample_size,)
    
    # 1. Test without JIT (as a baseline)
    print("\nTesting without JIT...")
    start_time = time.time()
    for i in range(args.iterations):
        subkey = random.fold_in(key, i)
        samples = sample_von_mises(subkey, loc, concentration, shape=shape)
        # Force computation to complete
        _ = samples.block_until_ready()
    non_jit_time = time.time() - start_time
    print(f"Time without JIT: {non_jit_time:.4f} seconds")
    print(f"Samples per second: {args.sample_size * args.iterations / non_jit_time:.0f}")
    
    # 2. Test with JIT but INCORRECT static_argnums (to demonstrate error)
    print("\nTesting with INCORRECT JIT (missing static_argnums)...")
    try:
        incorrect_jitted = jax.jit(sample_von_mises)  # Missing static_argnums
        # This will raise an error
        _ = incorrect_jitted(key, loc, concentration, shape=shape)
        print("ERROR: This should have failed but didn't.")
    except Exception as e:
        print(f"Got expected error: {type(e).__name__}")
        print("This is normal - JIT requires static_argnums for shape parameters.")
    
    # 3. Test with JIT with CORRECT static_argnums
    print("\nTesting with CORRECT JIT (using static_argnums)...")
    jitted_sampler = jax.jit(sample_von_mises, static_argnums=(3,))
    
    # First call includes compilation time
    print("First call (includes compilation)...")
    start_time = time.time()
    samples = jitted_sampler(key, loc, concentration, shape=shape)
    _ = samples.block_until_ready()
    first_call_time = time.time() - start_time
    print(f"First call time: {first_call_time:.4f} seconds")
    
    # Subsequent calls (should be much faster)
    print("Subsequent calls...")
    start_time = time.time()
    for i in range(args.iterations):
        subkey = random.fold_in(key, i)
        samples = jitted_sampler(subkey, loc, concentration, shape=shape)
        _ = samples.block_until_ready()
    jit_time = time.time() - start_time
    print(f"Time with JIT: {jit_time:.4f} seconds")
    print(f"Samples per second: {args.sample_size * args.iterations / jit_time:.0f}")
    
    # Calculate speedup
    speedup = non_jit_time / jit_time
    print(f"\nJIT speedup: {speedup:.1f}x")
    
    # 4. Test with alternative static_argnames
    print("\nTesting with static_argnames...")
    jitted_sampler_named = jax.jit(sample_von_mises, static_argnames=('shape',))
    
    start_time = time.time()
    for i in range(args.iterations):
        subkey = random.fold_in(key, i)
        samples = jitted_sampler_named(subkey, loc, concentration, shape=shape)
        _ = samples.block_until_ready()
    named_jit_time = time.time() - start_time
    print(f"Time with static_argnames: {named_jit_time:.4f} seconds")
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"Non-JIT time:     {non_jit_time:.4f} s")
    print(f"JIT first call:   {first_call_time:.4f} s (includes compilation)")
    print(f"JIT time:         {jit_time:.4f} s")
    print(f"JIT speedup:      {speedup:.1f}x")
    
    if speedup > 10:
        print("\nSUCCESS: JIT compilation is working correctly and providing a significant speedup!")
    else:
        print("\nWARNING: JIT speedup is lower than expected. This might be due to:")
        print("  - Small sample size (try increasing --sample-size)")
        print("  - Hardware limitations")
        print("  - Other overhead in the sampling process")


if __name__ == "__main__":
    main() 