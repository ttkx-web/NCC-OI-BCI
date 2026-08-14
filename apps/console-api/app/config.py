from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_NEURACLE_JELLYFISH_HOST = "127.0.0.1"
DEFAULT_NEURACLE_JELLYFISH_PORT = 8712


def _jellyfish_host_from_env() -> str:
    value = os.environ.get("NEURACLE_JELLYFISH_HOST")
    if value is None:
        return DEFAULT_NEURACLE_JELLYFISH_HOST
    host = value.strip()
    if not host:
        raise ValueError("NEURACLE_JELLYFISH_HOST must be non-empty")
    return host


def _jellyfish_port_from_env() -> int:
    value = os.environ.get("NEURACLE_JELLYFISH_PORT")
    if value is None:
        return DEFAULT_NEURACLE_JELLYFISH_PORT
    try:
        port = int(value)
    except ValueError as error:
        raise ValueError("NEURACLE_JELLYFISH_PORT must be an integer") from error
    if not 1 <= port <= 65535:
        raise ValueError("NEURACLE_JELLYFISH_PORT must be in 1..65535")
    return port


@dataclass(frozen=True, slots=True)
class Settings:
    repository_root: Path = Path(__file__).resolve().parents[3]
    api_prefix: str = "/api/v1"
    websocket_prefix: str = "/ws/v1"
    neuracle_jellyfish_host: str = field(default_factory=_jellyfish_host_from_env)
    neuracle_jellyfish_port: int = field(default_factory=_jellyfish_port_from_env)

    @property
    def model_roots(self) -> tuple[Path, ...]:
        return (self.repository_root / "model_packages", self.repository_root / "runs")

    @property
    def dataset_root(self) -> Path:
        return self.repository_root / "data" / "processed"


settings = Settings()
