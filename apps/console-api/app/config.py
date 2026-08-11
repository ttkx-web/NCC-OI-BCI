from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    repository_root: Path = Path(__file__).resolve().parents[3]
    api_prefix: str = "/api/v1"
    websocket_prefix: str = "/ws/v1"

    @property
    def model_roots(self) -> tuple[Path, ...]:
        return (self.repository_root / "model_packages", self.repository_root / "runs")

    @property
    def dataset_root(self) -> Path:
        return self.repository_root / "data" / "processed"


settings = Settings()

