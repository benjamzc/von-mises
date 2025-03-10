"""
Simple profiling script for JAX von Mises sampling.

This script provides basic performance metrics without complex dependencies.
"""

import time
import sys
import os
import gc
import traceback
import argparse

import jax
import jax.numpy as jnp
from jax import random, jit, vmap

# Try to import memory profiling if available
try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

from jax_von_mises.sampler import sample_von_mises, vmises_log_prob, compute_p


def get_memory_usage():
    """Get current memory usage in MB if psutil is available."""
    if HAVE_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    return 0


def safe_block_until_ready(x):
    """Safely block until computation is ready, with timeout."""
    try:
        return x.block_until_ready()
    except:
        print("  Warning: block_until_ready failed, computation may be incomplete")
        return x


def print_step(step_name):
    """Print a step header with clear separation."""
    print(f"\n{'=' * 10} {step_name} {'=' * 10}")


def profile_basic_functions(args):
    """
    Profile the basic functions of the JAX von Mises library.
    
    Args:
        args: Command line arguments with profiling options
    """
    print("\n=== JAX von Mises Profiling ===")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    if HAVE_PSUTIL:
        print(f"Initial memory usage: {get_memory_usage():.1f} MB")
    print()
    
    key = random.PRNGKey(42)
    results = {}
    
    try:
        # STEP 1: Profile vmises_log_prob
        print_step("Testing vmises_log_prob")
        x = jnp.linspace(-jnp.pi, jnp.pi, 1000)
        vmises_log_prob_jit = jit(vmises_log_prob)
        
        # Warm up
        _ = safe_block_until_ready(vmises_log_prob_jit(x, 0.0, 5.0))
        
        # Time execution
        start = time.time()
        for i in range(100):
            if i % 20 == 0 and i > 0:
                print(f"  Progress: {i}/100 iterations")
            _ = safe_block_until_ready(vmises_log_prob_jit(x, 0.0, 5.0))
        elapsed = time.time() - start
        
        ms_per_call = elapsed/100*1000
        print(f"  Time: {elapsed:.4f}s for 100 iterations ({ms_per_call:.2f}ms per call)")
        results['vmises_log_prob'] = ms_per_call
        
        # Free memory
        x = None
        gc.collect()
        
        # STEP 2: Profile compute_p
        print_step("Testing compute_p")
        kappa_values = jnp.logspace(-3, 3, 1000)
        compute_p_jit = jit(compute_p)
        
        # Warm up
        _ = safe_block_until_ready(compute_p_jit(kappa_values))
        
        # Time execution
        start = time.time()
        for i in range(100):
            if i % 20 == 0 and i > 0:
                print(f"  Progress: {i}/100 iterations")
            _ = safe_block_until_ready(compute_p_jit(kappa_values))
        elapsed = time.time() - start
        
        ms_per_call = elapsed/100*1000
        print(f"  Time: {elapsed:.4f}s for 100 iterations ({ms_per_call:.2f}ms per call)")
        results['compute_p'] = ms_per_call
        
        # Free memory
        kappa_values = None
        gc.collect()
        
        # STEP 3: Profile sample_von_mises with JIT
        print_step("Testing sample_von_mises with JIT")
        sample_jit = jit(sample_von_mises, static_argnums=(3,))
        shape = (args.sample_size,)
        
        print(f"  Using sample size: {args.sample_size}")
        
        try:
            # Warm up with smaller size first
            warmup_shape = (min(1000, args.sample_size),)
            _ = safe_block_until_ready(sample_jit(key, 0.0, 5.0, warmup_shape))
            print("  Warmup complete")
            
            # Time execution
            start = time.time()
            for i in range(10):
                print(f"  Progress: iteration {i+1}/10")
                _ = safe_block_until_ready(sample_jit(key, 0.0, 5.0, shape))
                if HAVE_PSUTIL:
                    print(f"  Memory: {get_memory_usage():.1f} MB")
            elapsed = time.time() - start
            
            ms_per_call = elapsed/10*1000
            samples_per_second = args.sample_size/(elapsed/10)
            print(f"  Time: {elapsed:.4f}s for 10 iterations ({ms_per_call:.2f}ms per call)")
            print(f"  Samples per second: {samples_per_second:.0f}")
            results['sample_jit_time'] = ms_per_call
            results['sample_jit_throughput'] = samples_per_second
            
            # Free memory
            gc.collect()
            
            # STEP 4: Profile without JIT (only if requested)
            if not args.skip_non_jit:
                print_step("Testing sample_von_mises without JIT")
                print("  This may be slow...")
                
                # Reduce iterations for non-JIT version
                non_jit_iters = 3
                
                start = time.time()
                for i in range(non_jit_iters):
                    print(f"  Progress: iteration {i+1}/{non_jit_iters}")
                    _ = safe_block_until_ready(sample_von_mises(key, 0.0, 5.0, shape=shape))
                    if HAVE_PSUTIL:
                        print(f"  Memory: {get_memory_usage():.1f} MB")
                elapsed = time.time() - start
                
                ms_per_call = elapsed/non_jit_iters*1000
                samples_per_second = args.sample_size/(elapsed/non_jit_iters)
                print(f"  Time: {elapsed:.4f}s for {non_jit_iters} iterations ({ms_per_call:.2f}ms per call)")
                print(f"  Samples per second: {samples_per_second:.0f}")
                
                # Calculate speedup
                jit_time = results['sample_jit_time'] / 1000  # Convert back to seconds
                non_jit_time = ms_per_call / 1000
                speedup = non_jit_time / jit_time
                print(f"  JIT speedup: {speedup:.1f}x")
                results['jit_speedup'] = speedup
                
                # Free memory
                gc.collect()
            else:
                print("\nSkipping non-JIT testing (use --all to enable)")
            
            # STEP 5: Profile with different concentration values
            print_step("Testing with different concentration values")
            
            for kappa in [0.1, 1.0, 10.0, 100.0]:
                print(f"  Testing κ={kappa}...")
                start = time.time()
                for i in range(5):
                    _ = safe_block_until_ready(sample_jit(key, 0.0, kappa, shape))
                elapsed = time.time() - start
                
                samples_per_sec = args.sample_size/(elapsed/5)
                print(f"  κ={kappa}: {elapsed:.4f}s for 5 iterations ({samples_per_sec:.0f} samples/sec)")
                results[f'kappa_{kappa}'] = samples_per_sec
                
                # Free memory
                gc.collect()
            
            # STEP 6: Profile vmap (only for smaller sample sizes)
            if args.sample_size <= 10000 and not args.skip_vmap:
                print_step("Testing batch processing with vmap")
                
                # Define a simple sampling function
                def sample_fn(key, loc, concentration):
                    item_shape = (100,)
                    return sample_von_mises(key, loc, concentration, shape=item_shape)
                
                # Create a batched version with vmap
                batch_sample = vmap(sample_fn)
                batch_sample_jit = jit(batch_sample)
                
                # Test with different batch sizes
                for batch_size in [10, 50]:
                    print(f"  Testing batch size {batch_size}...")
                    
                    # Create batch inputs
                    sub_key = random.fold_in(key, batch_size)
                    keys = random.split(sub_key, batch_size)
                    locs = jnp.zeros(batch_size)
                    concentrations = jnp.ones(batch_size) * 5.0
                    
                    # Warm up
                    print("  Warming up...")
                    _ = safe_block_until_ready(batch_sample_jit(keys, locs, concentrations))
                    
                    # Time with vmap+jit
                    print("  Timing vmap...")
                    start = time.time()
                    _ = safe_block_until_ready(batch_sample_jit(keys, locs, concentrations))
                    vmap_time = time.time() - start
                    
                    # Time without vmap (sequential) - only for small batches
                    if batch_size <= 20:
                        print("  Timing sequential...")
                        start = time.time()
                        samples = []
                        for i in range(batch_size):
                            samples.append(sample_fn(keys[i], locs[i], concentrations[i]))
                            if i % 5 == 0 and i > 0:
                                print(f"    Sequential progress: {i}/{batch_size}")
                        sequential_time = time.time() - start
                        speedup = sequential_time / vmap_time
                    else:
                        speedup = "N/A"
                    
                    total_samples = batch_size * 100
                    print(f"  Batch size {batch_size}: {vmap_time:.4f}s ({total_samples/vmap_time:.0f} samples/sec), Speedup: {speedup}")
                    results[f'vmap_batch_{batch_size}'] = total_samples/vmap_time
                    
                    # Free memory
                    keys = locs = concentrations = None
                    gc.collect()
            else:
                print("\nSkipping vmap testing (large sample size or disabled)")
            
        except Exception as e:
            print(f"Error during profiling: {e}")
            traceback.print_exc()
            print("\nContinuing with next test...")
            
    except Exception as e:
        print(f"Error during profiling: {e}")
        traceback.print_exc()
    
    finally:
        # Print summary
        print("\n=== Summary ===")
        for name, value in results.items():
            print(f"{name}: {value}")
        
        if HAVE_PSUTIL:
            print(f"Final memory usage: {get_memory_usage():.1f} MB")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Profile JAX von Mises sampling.')
    parser.add_argument('--sample-size', type=int, default=1000,
                        help='Sample size for testing (default: 1000)')
    parser.add_argument('--skip-non-jit', action='store_true',
                        help='Skip the non-JIT performance testing (which can be very slow)')
    parser.add_argument('--skip-vmap', action='store_true',
                        help='Skip vmap testing')
    parser.add_argument('--all', action='store_true',
                        help='Run all tests including slow ones')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.all:
        args.skip_non_jit = False
        args.skip_vmap = False
    profile_basic_functions(args) 