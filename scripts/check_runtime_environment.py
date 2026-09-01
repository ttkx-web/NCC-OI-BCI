"""Report deployment-safe Python and CUDA runtime metadata.

This probe never opens an EEG source, loads a Runtime Package, or stores
hardware serial numbers.  It performs one small CUDA tensor operation only
when CUDA is available.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import platform
import sys
from typing import Any


def _distribution_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def collect_environment(
    *,
    torch_module: Any | None = None,
) -> dict[str, object]:
    """Collect non-identifying runtime metadata without touching devices."""
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as error:
            torch_module = None
            torch_error: str | None = str(error)
        else:
            torch_error = None
    else:
        torch_error = None

    report: dict[str, object] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy_version": _distribution_version("numpy"),
        "scipy_version": _distribution_version("scipy"),
        "mne_version": _distribution_version("mne"),
        "torch_version": None,
        "torch_cuda_runtime": None,
        "cuda_available": False,
        "gpu_name": None,
        "compute_capability": None,
        "cuda_tensor_smoke": "not_run",
    }

    if torch_module is None:
        report["torch_import_error"] = torch_error
        return report

    report["torch_version"] = str(torch_module.__version__)
    report["torch_cuda_runtime"] = getattr(
        torch_module.version,
        "cuda",
        None,
    )
    cuda_available = bool(torch_module.cuda.is_available())
    report["cuda_available"] = cuda_available
    if not cuda_available:
        return report

    report["gpu_name"] = str(torch_module.cuda.get_device_name(0))
    capability = torch_module.cuda.get_device_capability(0)
    report["compute_capability"] = [
        int(capability[0]),
        int(capability[1]),
    ]
    try:
        value = (
            torch_module.tensor([1.0, 2.0], device="cuda")
            * 2.0
        ).sum().item()
    except Exception as error:  # pragma: no cover - hardware-specific path
        report["cuda_tensor_smoke"] = "failed"
        report["cuda_tensor_smoke_error"] = type(error).__name__
    else:
        report["cuda_tensor_smoke"] = "passed"
        report["cuda_tensor_smoke_value"] = float(value)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print non-identifying NCC-OI-BCI runtime metadata.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit non-zero unless CUDA and the tiny tensor smoke pass.",
    )
    arguments = parser.parse_args(argv)
    report = collect_environment()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if arguments.require_cuda and (
        not report["cuda_available"]
        or report["cuda_tensor_smoke"] != "passed"
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
