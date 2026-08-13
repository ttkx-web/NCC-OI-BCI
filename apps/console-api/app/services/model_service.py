from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from bci_dayloop.packages.loader import load_runtime_package

from app.schemas.models import ModelSummary


logger = logging.getLogger(__name__)

SUPPORTED_RUNTIME_TYPES = {"model_50m", "labram", "cbramod"}
DISPLAY_NAMES = {"model_50m": "50M", "labram": "LaBraM", "cbramod": "CBraMod"}


@dataclass(frozen=True, slots=True)
class ModelEntry:
    summary: ModelSummary
    package_path: Path
    payload: dict[str, Any]


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"package.yaml field {key!r} must be a mapping")
    return value


def _subject_id(path: Path, offline: dict[str, Any]) -> str | None:
    raw = offline.get("subject_id")
    if raw not in (None, ""):
        text = str(raw).upper()
        return text if text.startswith("S") else f"S{int(text):02d}" if text.isdigit() else text
    for part in path.parts:
        match = re.fullmatch(r"subject[_-]?(\d+)", part, re.IGNORECASE)
        if match:
            return f"S{int(match.group(1)):02d}"
    return None


def _metric(metrics: dict[str, Any], name: str) -> float | None:
    candidates: Iterable[Any] = (
        metrics.get("final_test"),
        metrics.get("target_final_test", {}).get("metrics") if isinstance(metrics.get("target_final_test"), dict) else None,
        metrics.get("population_validation", {}).get("best_metrics") if isinstance(metrics.get("population_validation"), dict) else None,
    )
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get(name) is not None:
            try:
                return float(candidate[name])
            except (TypeError, ValueError):
                return None
    return None


def _safe_file(package_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Runtime Package file entry must be a non-empty string")
    candidate = (package_path / value).resolve()
    if package_path.resolve() not in candidate.parents:
        raise ValueError("Runtime Package file entry escapes its package directory")
    return candidate


class ModelRegistry:
    def __init__(self, roots: Iterable[Path], *, runtime_verifier: Any | None = None) -> None:
        self.roots = tuple(Path(root).resolve() for root in roots)
        self._entries: dict[str, ModelEntry] = {}
        self._runtime_verifier = runtime_verifier or self._verify_runtime
        self._verification_cache: dict[Path, bool] = {}

    def _diagnostic_path(self, path: Path) -> str:
        for root in self.roots:
            try:
                return path.relative_to(root).as_posix()
            except ValueError:
                continue
        return path.name

    def _diagnostic_reason(self, error: Exception) -> str:
        reason = str(error)
        for root in self.roots:
            reason = reason.replace(str(root), "<model-root>")
        return reason

    def _verify_runtime(self, package_path: Path) -> bool:
        """Use the sole Runtime Package loader; never infer validity from YAML."""
        cached = self._verification_cache.get(package_path)
        if cached is not None:
            return cached
        try:
            load_runtime_package(package_path, device="cpu", verify_hashes=True)
        except Exception as error:
            logger.warning(
                "runtime package verification failed package=%s reason=%s: %s",
                self._diagnostic_path(package_path),
                type(error).__name__,
                self._diagnostic_reason(error),
            )
            verified = False
        else:
            verified = True
        self._verification_cache[package_path] = verified
        return verified

    def refresh(self) -> None:
        entries: dict[str, ModelEntry] = {}
        for root in self.roots:
            if not root.is_dir():
                continue
            for manifest in root.rglob("package.yaml"):
                try:
                    entry = self._read(root, manifest)
                except (OSError, TypeError, ValueError, yaml.YAMLError, json.JSONDecodeError) as error:
                    logger.warning(
                        "skipped runtime package manifest=%s reason=%s: %s",
                        self._diagnostic_path(manifest),
                        type(error).__name__,
                        self._diagnostic_reason(error),
                    )
                    continue
                entries[entry.summary.id] = entry
        self._entries = entries

    def _read(self, root: Path, manifest: Path) -> ModelEntry:
        package_path = manifest.parent.resolve()
        payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 2:
            raise ValueError("Unsupported Runtime Package schema")
        package = _mapping(payload, "package")
        model = _mapping(payload, "model")
        contract = _mapping(payload, "input_contract")
        runtime = _mapping(payload, "runtime")
        files = _mapping(payload, "files")
        adaptation = payload.get("adaptation") if isinstance(payload.get("adaptation"), dict) else {}
        offline = adaptation.get("offline") if isinstance(adaptation.get("offline"), dict) else {}

        required_keys = ("backbone", "classifier", "preprocessing", "metrics")
        required_paths = [_safe_file(package_path, files.get(key)) for key in required_keys]
        if not all(path.is_file() for path in required_paths):
            raise ValueError("Runtime Package is incomplete")
        metrics = json.loads(required_paths[-1].read_text(encoding="utf-8"))
        if not isinstance(metrics, dict):
            raise ValueError("metrics.json must contain an object")

        package_id = str(package.get("id", "")).strip()
        version = str(package.get("version", "v1")).strip()
        if not package_id:
            raise ValueError("Runtime Package id is required")
        relative_key = package_path.relative_to(root).as_posix().lower()
        digest = hashlib.sha256(f"{package_id}|{version}|{relative_key}".encode()).hexdigest()[:16]
        model_id = f"model_{digest}"
        model_type = str(model.get("type", "")).lower()
        channel_names = contract.get("channel_names")
        class_names = model.get("class_names")
        if not isinstance(channel_names, list) or not channel_names or not isinstance(class_names, list) or not class_names:
            raise ValueError("Runtime Package channel and class contracts are required")
        subject_id = _subject_id(package_path, offline)
        summary = ModelSummary(
            id=model_id,
            model_name=DISPLAY_NAMES.get(model_type, str(model.get("name", model_type))),
            model_type=model_type,
            head_type=str(offline.get("head_type", "population")).lower(),
            subject_id=subject_id,
            dataset_name=str(model.get("dataset", "unknown")),
            task=str(model.get("task", "unknown")),
            window_sec=float(contract["window_sec"]),
            step_sec=float(runtime.get("step_sec", 0.5)),
            sample_rate=float(contract["sample_rate"]),
            target_channels=len(channel_names),
            schema_version=2,
            runtime_verified=(model_type in SUPPORTED_RUNTIME_TYPES and bool(self._runtime_verifier(package_path))),
            package_version=version,
            balanced_accuracy=_metric(metrics, "balanced_accuracy"),
            macro_f1=_metric(metrics, "macro_f1"),
            warning_message=package.get("warning_message"),
        )
        return ModelEntry(summary=summary, package_path=package_path, payload=payload)

    def list(self, *, backbone: str | None = None, subject: str | None = None, adaptation: str | None = None) -> list[ModelSummary]:
        self.refresh()
        items = [entry.summary for entry in self._entries.values()]
        if backbone:
            query = backbone.lower().replace("-", "_")
            items = [item for item in items if query in {item.model_type, item.model_name.lower()}]
        if subject:
            items = [item for item in items if (item.subject_id or "").lower() == subject.lower()]
        if adaptation:
            items = [item for item in items if item.head_type == adaptation.lower()]
        return sorted(items, key=lambda item: (item.model_name, item.head_type, item.subject_id or ""))

    def get_entry(self, model_id: str, *, refresh: bool = True) -> ModelEntry:
        if refresh:
            self.refresh()
        try:
            return self._entries[model_id]
        except KeyError as error:
            raise LookupError(f"Unknown model_id: {model_id}") from error
