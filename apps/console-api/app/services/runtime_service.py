from __future__ import annotations


def runtime_available() -> bool:
    try:
        from bci_dayloop.inference.runtime_control import PipelineController  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def system_status() -> dict[str, object]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except (ImportError, OSError):
        cuda_available = False
    return {
        "device": {"status": "disconnected", "source": None},
        "runtime": {"state": "idle"},
        "compute": {"cuda_available": cuda_available},
    }
