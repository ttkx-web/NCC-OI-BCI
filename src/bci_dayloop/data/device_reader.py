"""Interfaces for offline EEG device-file readers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from bci_dayloop.data.records import RawEEGRecord


@runtime_checkable
class DeviceReader(Protocol):
    """Load one offline device recording into a validated raw record."""

    reader_name: str

    def load(
        self,
        path: str | Path,
        *,
        subject_id: str | None = None,
        session_id: str | None = None,
        device_id: str | None = None,
    ) -> RawEEGRecord:
        """Load a recording without imposing a realtime acquisition interface."""


ReaderBuilder = Callable[..., DeviceReader]


class DeviceReaderFactory:
    """Explicit registry for offline device readers."""

    _registry: ClassVar[dict[str, ReaderBuilder]] = {}

    @staticmethod
    def _normalize_name(name: str) -> str:
        if not isinstance(name, str):
            raise ValueError("Reader name must be a non-empty string")
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("Reader name must be non-empty")
        return normalized

    @classmethod
    def register(cls, name: str, builder: ReaderBuilder) -> None:
        normalized = cls._normalize_name(name)
        if normalized in cls._registry:
            raise ValueError(f"Reader already registered: {normalized}")
        cls._registry[normalized] = builder

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> DeviceReader:
        normalized = cls._normalize_name(name)
        try:
            builder = cls._registry[normalized]
        except KeyError as exc:
            available = ", ".join(cls.list_readers()) or "(none)"
            raise ValueError(
                f"Unknown device reader: {normalized}. Available readers: {available}"
            ) from exc
        return builder(**kwargs)

    @classmethod
    def list_readers(cls) -> list[str]:
        return sorted(cls._registry)


def register_builtin_readers() -> None:
    """Register built-in offline readers without overwriting existing registrations."""
    if "neuracle-bdf" not in DeviceReaderFactory._registry:
        from bci_dayloop.data.neuracle_bdf import NeuracleBDFReader

        DeviceReaderFactory.register(NeuracleBDFReader.reader_name, NeuracleBDFReader)
