"""
Genesis GPU Configuration Module.

This module ensures Genesis uses GPU acceleration by default.
Import this before genesis, or use configure_gpu() after importing genesis.
"""

import os
import sys
import subprocess


def detect_gpu():
    """Detect available GPU and return the appropriate Genesis backend string."""
    # Check NVIDIA GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_info = result.stdout.strip().split("\n")[0]
            print(f"[Genesis GPU Config] Detected NVIDIA GPU: {gpu_info}")
            return "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check AMD GPU
    try:
        result = subprocess.run(["rocm-smi", "--showid"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            print("[Genesis GPU Config] Detected AMD GPU via ROCm")
            return "amdgpu"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check Apple Metal
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "Metal" in result.stdout:
                print("[Genesis GPU Config] Detected Apple Metal GPU")
                return "metal"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Check via PyTorch
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[Genesis GPU Config] Detected CUDA GPU via PyTorch: {gpu_name}")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[Genesis GPU Config] Detected Apple Metal GPU via PyTorch")
            return "metal"
    except ImportError:
        pass

    print("[Genesis GPU Config] No GPU detected, falling back to CPU")
    return "cpu"


def configure_gpu():
    """Configure Genesis to use GPU acceleration."""
    backend_name = detect_gpu()

    # Set environment variables for optimal GPU performance
    if backend_name == "cuda":
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "")
    elif backend_name == "amdgpu":
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "")

    return backend_name


def init_genesis_gpu(**kwargs):
    """Initialize Genesis with GPU backend and return the backend used.

    All kwargs are passed through to gs.init() (e.g. precision, theme, seed).
    """
    import genesis as gs

    backend_name = configure_gpu()
    backend_map = {
        "cuda": gs.cuda,
        "amdgpu": gs.amdgpu,
        "metal": gs.metal,
        "cpu": gs.cpu,
    }
    backend = backend_map.get(backend_name, gs.gpu)
    kwargs.setdefault("backend", backend)
    gs.init(**kwargs)
    return backend_name


if __name__ == "__main__":
    backend = detect_gpu()
    print(f"\nRecommended Genesis backend: {backend}")
    print(f"\nUsage in your scripts:")
    print(f"  import genesis as gs")
    print(f"  gs.init(backend=gs.{backend})")
