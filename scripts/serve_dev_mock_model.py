from __future__ import annotations

import argparse

from _bootstrap import ROOT  # noqa: F401
from bci_dayloop.serving.backends import MockModelBackend
from bci_dayloop.serving.profiles import DEVICE_PROFILES, get_device_profile
from bci_dayloop.serving.server import run_model_service


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dev mock model WebSocket for Passive BCI window replay."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument(
        "--profile",
        default="neuracle59",
        choices=sorted(DEVICE_PROFILES),
        help="Advertised hello window layout (mock accepts neuracle59 and bcigo32 live streams).",
    )
    args = parser.parse_args()
    profile = get_device_profile(args.profile)
    backend = MockModelBackend(profile=profile)
    print(
        f"[model-service] mock hello window={profile.window_sec:g}s "
        f"(accepts neuracle59 / bcigo32 live layouts; --profile only sets advertised window)"
    )
    run_model_service(backend, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
