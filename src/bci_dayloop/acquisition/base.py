from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

EEGChunk: TypeAlias = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True, slots=True)
class AcquirerMetadata:
    name: str
    sample_rate: float
    channel_names: list[str]
    unit: str

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)


class AbstractAcquirer(ABC):
    metadata: AcquirerMetadata

    @abstractmethod
    def start_stream(self) -> None:
        """Start acquisition or replay."""

    @abstractmethod
    def stop_stream(self) -> None:
        """Stop acquisition or replay."""

    @abstractmethod
    def get_chunk(self, window_sec: float | None = None) -> EEGChunk:
        """Return the latest complete window and timestamps."""

    @abstractmethod
    def get_new_samples(self) -> EEGChunk:
        """Return samples that arrived since the previous call."""

