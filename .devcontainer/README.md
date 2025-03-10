# Development Container Configurations

This directory contains development container configurations for the JAX Von Mises library, supporting both NVIDIA (CUDA) and AMD (ROCm) GPUs.

## Available Configurations

- **Default (CPU)**: Basic configuration for CPU-only development
- **NVIDIA GPU**: Configuration for systems with NVIDIA GPUs using CUDA
- **AMD GPU**: Configuration for systems with AMD GPUs using ROCm

## Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop) installed on your machine
- [Visual Studio Code](https://code.visualstudio.com/) with the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension installed
- For NVIDIA GPU support: NVIDIA Docker runtime and NVIDIA drivers installed
- For AMD GPU support: ROCm drivers installed (see [ROCm installation guide](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html))

## Using the Configurations

### For NVIDIA GPUs

1. Open this repository in Visual Studio Code
2. Open the Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
3. Select `Remote-Containers: Rebuild and Reopen in Container`
4. Choose the "JAX Von Mises (NVIDIA GPU)" variant

### For AMD GPUs with ROCm

1. Open this repository in Visual Studio Code
2. Open the Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
3. Select `Remote-Containers: Rebuild and Reopen in Container`
4. Choose the "JAX Von Mises (AMD GPU)" variant

## Verifying GPU Support

After opening the container, you can verify that your GPU is properly detected:

```python
import jax
print(jax.devices())
```

For NVIDIA GPUs, you should see device outputs like `CudaDevice(id=0)`.
For AMD GPUs, you should see device outputs like `RocmDevice(id=0)`.

## ROCm Specific Notes

The AMD GPU configuration uses the official `rocm/jax-community` Docker image, which comes with:

- ROCm libraries pre-installed
- JAX with ROCm support configured

The container is configured with the following important settings:

- `--device=/dev/kfd` and `--device=/dev/dri`: Provides access to the AMD GPU devices
- `--group-add=video`: Adds the user to the video group for GPU access
- `--network=host`: Simplifies network access
- `--shm-size=64G`: Allocates shared memory for efficient GPU operations

### ROCm Environment Variables

The container automatically sets:

```
LLVM_PATH=/opt/rocm/llvm
```

This environment variable helps JAX find the LLVM tools at runtime.

## Troubleshooting

### Common Issues with AMD GPUs/ROCm

1. **GPU not detected**: Ensure your ROCm drivers are properly installed. Check with `rocm-smi` on the host.
2. **Permission errors**: Make sure your user has proper permissions for `/dev/kfd` and `/dev/dri`.
3. **Shared memory errors**: Adjust the `--shm-size` parameter if needed for your workload.

For more details, consult the [JAX on ROCm documentation](https://rocm.docs.amd.com/projects/jax/en/latest/). 