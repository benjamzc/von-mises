"""Benchmarking and profiling tools for JAX von Mises sampling."""

from benchmarks.performance_benchmark import (
    benchmark_jit_vs_nojit,
    benchmark_concentration_impact,
    benchmark_vmap_scaling,
    benchmark_pmap_multi_device,
    benchmark_neural_network,
    profile_memory_usage,
    plot_benchmark_results,
)

from benchmarks.profile_sampler import (
    profile_main_functions,
    profile_batch_processing,
    profile_memory_leaks,
    profile_gpu_utilization,
    run_all_profiles,
)

__all__ = [
    "benchmark_jit_vs_nojit",
    "benchmark_concentration_impact",
    "benchmark_vmap_scaling",
    "benchmark_pmap_multi_device",
    "benchmark_neural_network",
    "profile_memory_usage",
    "plot_benchmark_results",
    "profile_main_functions",
    "profile_batch_processing",
    "profile_memory_leaks",
    "profile_gpu_utilization", 
    "run_all_profiles",
] 