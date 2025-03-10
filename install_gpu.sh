#!/bin/bash
set -e

echo "Installing JAX with GPU (CUDA) support..."

# Determine CUDA version
if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    CUDA_VERSION_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_VERSION_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    echo "Detected CUDA version: $CUDA_VERSION"
else
    echo "CUDA not found. Please install CUDA toolkit first."
    exit 1
fi

# Install appropriate JAX version based on CUDA version
if [ "$CUDA_VERSION_MAJOR" -eq 11 ]; then
    if [ "$CUDA_VERSION_MINOR" -ge 8 ]; then
        echo "Installing JAX with CUDA 11.8+ support..."
        pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    else
        echo "Installing JAX with CUDA 11.x support..."
        pip install --upgrade "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
    fi
elif [ "$CUDA_VERSION_MAJOR" -eq 12 ]; then
    echo "Installing JAX with CUDA 12.x support..."
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
else
    echo "Unsupported CUDA version: $CUDA_VERSION. Please use CUDA 11.x or 12.x."
    exit 1
fi

# Install the package in development mode
pip install -e ".[dev,benchmark]"

echo "Installation complete. Please run 'python -m benchmarks.simple_profile' to verify GPU support." 