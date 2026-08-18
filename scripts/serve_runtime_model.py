from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from bci_dayloop.packages.loader import load_runtime_package
from bci_dayloop.serving.backends import RuntimePackageBackend
from bci_dayloop.serving.server import run_model_service


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve a Runtime Package over WebSocket for Passive BCI."
    )
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--strategy", default="none")
    parser.add_argument(
        "--state-file",
        default=None,
        help="Reserved for online-head state. Only --strategy none is implemented.",
    )
    args = parser.parse_args()
    if args.strategy != "none":
        raise SystemExit(
            f"Unsupported --strategy {args.strategy!r}. "
            "Passive BCI currently supports --strategy none."
        )
    package = load_runtime_package(args.package, device=args.device)
    print(
        f"[model-service] runtime package={package.package_path.name} "
        f"model={package.model_name} type={package.model_type} device={args.device}"
    )
    run_model_service(
        RuntimePackageBackend(package=package, strategy=args.strategy),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
