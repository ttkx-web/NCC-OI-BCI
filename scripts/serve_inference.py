from __future__ import annotations

"""Run the NCC-OI-BCI one-window inference service on localhost."""

import argparse
import logging
from pathlib import Path

from _bootstrap import ROOT
from bci_dayloop.inference.http_service import InferenceServiceRuntime, create_inference_server
from bci_dayloop.packages import load_multi_head_runtime_package
from bci_dayloop.applications.three_mental_states.contract import DEFAULT_PATHS


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve one complete [C,T] EEG window through the formal multi-head runtime package."
    )
    parser.add_argument("--model-package", default=DEFAULT_PATHS["model_package"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 <= args.port <= 65535:
        raise ValueError("port must be between 0 and 65535.")

    package_path = _path(args.model_package)
    # Loading happens exactly once, before the server accepts any HTTP request.
    predictor = load_multi_head_runtime_package(package_path, device=args.device)
    runtime = InferenceServiceRuntime(
        predictor=predictor,
        model_package=str(package_path),
        device=str(predictor.device),
    )
    server = create_inference_server(args.host, args.port, runtime)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(
        f"NCC-OI-BCI inference service ready at http://{args.host}:{server.server_port} "
        f"(package={package_path}, device={predictor.device})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping inference service.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
