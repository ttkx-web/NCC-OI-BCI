from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import h5py

from app.schemas.datasets import DatasetSummary


@dataclass(frozen=True, slots=True)
class DatasetEntry:
    summary: DatasetSummary
    path: Path


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "dataset"


class DatasetRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self._entries: dict[tuple[str, str], DatasetEntry] = {}

    def refresh(self) -> None:
        entries: dict[tuple[str, str], DatasetEntry] = {}
        if not self.root.is_dir():
            self._entries = entries
            return
        for pattern in ("*.h5", "*.hdf5"):
            for path in self.root.rglob(pattern):
                try:
                    entry = self._read(path.resolve())
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                entries[(entry.summary.id, entry.summary.subject_id.lower())] = entry
        self._entries = entries

    def _read(self, path: Path) -> DatasetEntry:
        with h5py.File(path, "r") as handle:
            dataset_name = str(handle.attrs["dataset_name"])
            channel_names = json.loads(handle.attrs["channel_names"])
            class_names = json.loads(handle.attrs["class_names"])
            sample_rate = float(handle.attrs["sample_rate"])
            unit = str(handle.attrs["unit"])
            sessions = sorted(set(handle["session_ids"].asstr()[:].tolist()))
            subject_values = sorted(set(int(value) for value in handle["subject_ids"][:].tolist()))
            trial_count = int(handle["data"].shape[0])
        if len(subject_values) != 1:
            raise ValueError("Console datasets must represent exactly one subject")
        subject_id = f"S{subject_values[0]:02d}"
        summary = DatasetSummary(
            id=_slug(dataset_name),
            name=dataset_name,
            subject_id=subject_id,
            sessions=sessions,
            trial_count=trial_count,
            channel_count=len(channel_names),
            sample_rate=sample_rate,
            unit=unit,
            class_names=[str(name) for name in class_names],
            qc_status="passed",
        )
        return DatasetEntry(summary=summary, path=path)

    def list(self) -> list[DatasetSummary]:
        self.refresh()
        return sorted((entry.summary for entry in self._entries.values()), key=lambda item: (item.name, item.subject_id))

    def get_entry(self, dataset_id: str, subject_id: str | None = None, *, refresh: bool = True) -> DatasetEntry:
        if refresh:
            self.refresh()
        candidates = [entry for (item_id, _), entry in self._entries.items() if item_id == dataset_id]
        if subject_id:
            candidates = [entry for entry in candidates if entry.summary.subject_id.lower() == subject_id.lower()]
        if len(candidates) != 1:
            detail = "subject_id is required" if len(candidates) > 1 else "dataset was not found"
            raise LookupError(f"{dataset_id}: {detail}")
        return candidates[0]
