"""
GPU detection and diagnostics for Genesis.

Detects available compute backends (CUDA, AMD GPU, Metal, CPU) and reports
hardware capabilities, driver versions, and compatibility with Genesis features.
"""

import os
import sys
import platform
import importlib
from dataclasses import dataclass, field


@dataclass
class DeviceInfo:
    """Information about a detected compute device."""

    name: str = "Unknown"
    backend: str = "cpu"
    device_index: int = 0
    total_memory_gb: float = 0.0
    driver_version: str = "N/A"
    compute_capability: str = "N/A"
    supports_64bit: bool = True
    supports_zerocopy: bool = False
    is_available: bool = False
    extra: dict = field(default_factory=dict)


def detect_all_devices() -> list[DeviceInfo]:
    """Detect all available compute devices and return their info."""
    devices = []

    # CPU is always available
    devices.append(_detect_cpu())

    # Try CUDA / AMD GPU (both use torch.cuda)
    devices.extend(_detect_cuda_devices())

    # Try Apple Metal
    metal = _detect_metal()
    if metal.is_available:
        devices.append(metal)

    return devices


def _detect_cpu() -> DeviceInfo:
    """Detect CPU info."""
    import psutil

    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
        name = info.get("brand_raw") or info.get("hardware_raw") or info.get("vendor_id_raw", "CPU")
    except Exception:
        name = platform.processor() or "CPU"

    return DeviceInfo(
        name=name,
        backend="cpu",
        device_index=0,
        total_memory_gb=psutil.virtual_memory().total / 1024**3,
        supports_64bit=True,
        supports_zerocopy=True,
        is_available=True,
        extra={"cores": os.cpu_count() or 1, "platform": platform.platform()},
    )


def _detect_cuda_devices() -> list[DeviceInfo]:
    """Detect CUDA or AMD GPU (HIP) devices via PyTorch."""
    devices = []
    try:
        import torch

        if not torch.cuda.is_available():
            return devices

        is_hip = bool(torch.version.hip)
        backend_name = "amdgpu" if is_hip else "cuda"
        driver = torch.version.hip if is_hip else (torch.version.cuda or "N/A")

        torch_version = tuple(map(int, torch.__version__.split(".")[:2]))

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            cc = f"{props.major}.{props.minor}" if not is_hip else "N/A"

            # Zero-copy requires compatible torch version
            supports_zc = True  # CUDA/HIP generally supports zero-copy

            devices.append(
                DeviceInfo(
                    name=props.name,
                    backend=backend_name,
                    device_index=i,
                    total_memory_gb=props.total_memory / 1024**3,
                    driver_version=driver,
                    compute_capability=cc,
                    supports_64bit=True,
                    supports_zerocopy=supports_zc,
                    is_available=True,
                    extra={
                        "multi_processor_count": props.multi_processor_count,
                        "torch_version": torch.__version__,
                    },
                )
            )
    except ImportError:
        pass
    except Exception:
        pass

    return devices


def _detect_metal() -> DeviceInfo:
    """Detect Apple Metal (MPS) device."""
    info = DeviceInfo(backend="metal")
    if sys.platform != "darwin":
        return info

    try:
        import torch

        if not torch.backends.mps.is_available():
            return info

        import psutil

        try:
            import cpuinfo

            cpu_info = cpuinfo.get_cpu_info()
            name = cpu_info.get("brand_raw", "Apple Silicon")
        except Exception:
            name = "Apple Silicon"

        torch_version_parts = torch.__version__.replace("+", ".").split(".")[:3]
        torch_version_tuple = tuple(map(int, torch_version_parts))
        supports_zc = torch_version_tuple > (2, 9, 1)

        info = DeviceInfo(
            name=name,
            backend="metal",
            device_index=0,
            total_memory_gb=psutil.virtual_memory().total / 1024**3,
            supports_64bit=False,  # Metal does not support 64-bit precision
            supports_zerocopy=supports_zc,
            is_available=True,
            extra={"torch_version": torch.__version__},
        )
    except ImportError:
        pass
    except Exception:
        pass

    return info


def get_recommended_backend(devices: list[DeviceInfo]) -> DeviceInfo:
    """Return the recommended device for Genesis based on priority: CUDA > AMD > Metal > CPU."""
    priority = {"cuda": 0, "amdgpu": 1, "metal": 2, "cpu": 3}
    available = [d for d in devices if d.is_available]
    available.sort(key=lambda d: priority.get(d.backend, 99))
    return available[0] if available else devices[0]


def check_genesis_dependencies() -> dict[str, dict]:
    """Check status of key Genesis dependencies."""
    deps = {
        "torch": {"required": True, "min_version": "2.8.0"},
        "quadrants": {"required": True, "min_version": "0.4.4"},
        "numpy": {"required": True, "min_version": None},
        "trimesh": {"required": True, "min_version": None},
        "mujoco": {"required": False, "min_version": "3.2.5"},
        "pyrender": {"required": True, "min_version": None},
        "pyglet": {"required": True, "min_version": None},
        "PIL": {"required": True, "min_version": None},
        "cv2": {"required": False, "min_version": None},
        "pydantic": {"required": True, "min_version": None},
    }

    results = {}
    for pkg, info in deps.items():
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            results[pkg] = {"installed": True, "version": version, **info}
        except ImportError:
            results[pkg] = {"installed": False, "version": None, **info}

    return results


def format_device_summary(device: DeviceInfo) -> str:
    """Format a device info into a human-readable summary string."""
    lines = [
        f"Device: {device.name}",
        f"Backend: {device.backend}",
        f"Memory: {device.total_memory_gb:.1f} GB",
        f"Driver: {device.driver_version}",
    ]
    if device.compute_capability != "N/A":
        lines.append(f"Compute Capability: {device.compute_capability}")
    lines.append(f"64-bit Precision: {'Yes' if device.supports_64bit else 'No'}")
    lines.append(f"Zero-Copy: {'Yes' if device.supports_zerocopy else 'No'}")
    return "\n".join(lines)
