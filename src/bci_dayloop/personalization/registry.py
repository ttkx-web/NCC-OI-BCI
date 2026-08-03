from __future__ import annotations

"""Persistent registry for user-specific model packages.

The registry stores package metadata and one active package per
``(user_id, task)``.  It never deletes model files automatically.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from .package import (
    LoadedPersonalModelPackage,
    load_personal_model_package,
    utc_now_iso,
    validate_personal_model_package,
)


RegistryStatus = Literal["candidate", "active", "archived"]


def _registry_key(user_id: str, task: str) -> str:
    return f"{user_id}::{task}"


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    package_id: str
    user_id: str
    task: str
    adaptation_type: str
    package_path: str
    runtime_ready: bool
    status: str
    created_at: str
    registered_at: str
    metrics: dict[str, Any]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "user_id": self.user_id,
            "task": self.task,
            "adaptation_type": self.adaptation_type,
            "package_path": self.package_path,
            "runtime_ready": bool(self.runtime_ready),
            "status": self.status,
            "created_at": self.created_at,
            "registered_at": self.registered_at,
            "metrics": self.metrics,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "RegistryEntry":
        return cls(
            package_id=str(payload["package_id"]),
            user_id=str(payload["user_id"]),
            task=str(payload["task"]),
            adaptation_type=str(payload["adaptation_type"]),
            package_path=str(payload["package_path"]),
            runtime_ready=bool(payload.get("runtime_ready", False)),
            status=str(payload.get("status", "candidate")),
            created_at=str(payload["created_at"]),
            registered_at=str(payload["registered_at"]),
            metrics=dict(payload.get("metrics", {})),
            notes=tuple(
                str(note) for note in payload.get("notes", [])
            ),
        )


class _RegistryLock:
    """Small portable lock-file implementation.

    It protects concurrent registry writes from CLI/UI processes.  Stale locks
    are removed after ``stale_seconds``.
    """

    def __init__(
        self,
        path: Path,
        *,
        timeout_seconds: float = 10.0,
        stale_seconds: float = 120.0,
    ) -> None:
        self.path = path
        self.timeout_seconds = float(timeout_seconds)
        self.stale_seconds = float(stale_seconds)
        self._acquired = False

    def __enter__(self) -> "_RegistryLock":
        started = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        while True:
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                    if age > self.stale_seconds:
                        self.path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue

                if time.monotonic() - started >= self.timeout_seconds:
                    raise TimeoutError(
                        f"Timed out waiting for registry lock: {self.path}"
                    )
                time.sleep(0.05)
                continue

            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "created_at": utc_now_iso(),
                        }
                    )
                )
            self._acquired = True
            return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._acquired:
            self.path.unlink(missing_ok=True)
            self._acquired = False


class PersonalModelRegistry:
    """JSON-backed registry for personal model packages."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        registry_path: str | Path,
        *,
        create: bool = True,
    ) -> None:
        self.path = Path(registry_path).expanduser().resolve()
        self.lock_path = Path(f"{self.path}.lock")
        if create and not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write_payload(self._empty_payload())

    def _empty_payload(self) -> dict[str, Any]:
        now = utc_now_iso()
        return {
            "format_version": self.FORMAT_VERSION,
            "created_at": now,
            "updated_at": now,
            "entries": {},
            "active": {},
        }

    def _read_payload(self) -> dict[str, Any]:
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Registry file was not found: {self.path}"
            )
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError("Registry root must be a JSON object.")
        if int(payload.get("format_version", -1)) != self.FORMAT_VERSION:
            raise ValueError(
                "Unsupported registry format_version: "
                f"{payload.get('format_version')!r}"
            )
        if not isinstance(payload.get("entries"), dict):
            raise TypeError("registry.entries must be a JSON object.")
        if not isinstance(payload.get("active"), dict):
            raise TypeError("registry.active must be a JSON object.")
        return dict(payload)

    def _write_payload(self, payload: Mapping[str, Any]) -> None:
        destination = self.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(f"{destination}.tmp")
        body = dict(payload)
        body["updated_at"] = utc_now_iso()
        temporary.write_text(
            json.dumps(
                body,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    def _stored_path(self, package_path: Path) -> str:
        try:
            return str(package_path.relative_to(self.path.parent))
        except ValueError:
            return str(package_path)

    def _resolved_path(self, stored_path: str) -> Path:
        value = Path(stored_path).expanduser()
        if value.is_absolute():
            return value.resolve()
        return (self.path.parent / value).resolve()

    def register(
        self,
        package_path: str | Path,
        *,
        status: RegistryStatus = "candidate",
        set_active: bool = False,
        replace: bool = False,
        validate: bool = True,
    ) -> RegistryEntry:
        if status not in {"candidate", "active", "archived"}:
            raise ValueError(f"Invalid registry status: {status!r}.")

        loaded = load_personal_model_package(
            package_path,
            validate=validate,
        )
        manifest = loaded.manifest
        metrics: dict[str, Any] = {}
        if manifest.metrics_path is not None:
            metrics_path = loaded.path / manifest.metrics_path
            with metrics_path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if isinstance(value, dict):
                metrics = dict(value)

        if set_active:
            status = "active"

        entry = RegistryEntry(
            package_id=manifest.package_id,
            user_id=manifest.user_id,
            task=manifest.task,
            adaptation_type=manifest.adaptation_type,
            package_path=self._stored_path(loaded.path),
            runtime_ready=manifest.runtime_ready,
            status=status,
            created_at=manifest.created_at,
            registered_at=utc_now_iso(),
            metrics=metrics,
            notes=manifest.notes,
        )

        with _RegistryLock(self.lock_path):
            payload = self._read_payload()
            entries = dict(payload["entries"])
            if entry.package_id in entries and not replace:
                raise KeyError(
                    f"Package ID is already registered: {entry.package_id}"
                )
            entries[entry.package_id] = entry.to_dict()
            payload["entries"] = entries

            if set_active:
                self._set_active_in_payload(
                    payload,
                    user_id=entry.user_id,
                    task=entry.task,
                    package_id=entry.package_id,
                )
            self._write_payload(payload)

        return self.get(entry.package_id)

    def _set_active_in_payload(
        self,
        payload: dict[str, Any],
        *,
        user_id: str,
        task: str,
        package_id: str,
    ) -> None:
        entries = dict(payload["entries"])
        if package_id not in entries:
            raise KeyError(
                f"Package ID is not registered: {package_id}"
            )
        selected = RegistryEntry.from_dict(entries[package_id])
        if selected.user_id != user_id or selected.task != task:
            raise ValueError(
                "The selected package belongs to a different user/task: "
                f"package=({selected.user_id}, {selected.task}), "
                f"requested=({user_id}, {task})."
            )
        if not selected.runtime_ready:
            raise ValueError(
                f"Package {package_id} is not runtime-ready and cannot "
                "be activated for CLI/Streamlit."
            )

        key = _registry_key(user_id, task)
        previous_id = payload["active"].get(key)
        if previous_id and previous_id in entries:
            previous = dict(entries[previous_id])
            if previous.get("status") == "active":
                previous["status"] = "candidate"
                entries[previous_id] = previous

        current = dict(entries[package_id])
        current["status"] = "active"
        entries[package_id] = current
        payload["entries"] = entries
        active = dict(payload["active"])
        active[key] = package_id
        payload["active"] = active

    def set_active(
        self,
        *,
        user_id: str,
        task: str,
        package_id: str,
    ) -> RegistryEntry:
        with _RegistryLock(self.lock_path):
            payload = self._read_payload()
            self._set_active_in_payload(
                payload,
                user_id=str(user_id),
                task=str(task),
                package_id=str(package_id),
            )
            self._write_payload(payload)
        return self.get(package_id)

    def clear_active(self, *, user_id: str, task: str) -> None:
        key = _registry_key(str(user_id), str(task))
        with _RegistryLock(self.lock_path):
            payload = self._read_payload()
            active = dict(payload["active"])
            package_id = active.pop(key, None)
            entries = dict(payload["entries"])
            if package_id in entries:
                entry = dict(entries[package_id])
                if entry.get("status") == "active":
                    entry["status"] = "candidate"
                    entries[package_id] = entry
            payload["entries"] = entries
            payload["active"] = active
            self._write_payload(payload)

    def get(self, package_id: str) -> RegistryEntry:
        payload = self._read_payload()
        entry = payload["entries"].get(str(package_id))
        if entry is None:
            raise KeyError(
                f"Package ID is not registered: {package_id}"
            )
        return RegistryEntry.from_dict(entry)

    def list_entries(
        self,
        *,
        user_id: str | None = None,
        task: str | None = None,
        status: str | None = None,
        runtime_ready: bool | None = None,
    ) -> list[RegistryEntry]:
        payload = self._read_payload()
        entries = [
            RegistryEntry.from_dict(value)
            for value in payload["entries"].values()
        ]

        def keep(entry: RegistryEntry) -> bool:
            if user_id is not None and entry.user_id != str(user_id):
                return False
            if task is not None and entry.task != str(task):
                return False
            if status is not None and entry.status != str(status):
                return False
            if (
                runtime_ready is not None
                and entry.runtime_ready != bool(runtime_ready)
            ):
                return False
            return True

        return sorted(
            [entry for entry in entries if keep(entry)],
            key=lambda entry: (
                entry.user_id,
                entry.task,
                entry.created_at,
                entry.package_id,
            ),
        )

    def get_active(
        self,
        *,
        user_id: str,
        task: str,
    ) -> RegistryEntry:
        payload = self._read_payload()
        key = _registry_key(str(user_id), str(task))
        package_id = payload["active"].get(key)
        if package_id is None:
            raise KeyError(
                f"No active package for user={user_id!r}, task={task!r}."
            )
        return RegistryEntry.from_dict(
            payload["entries"][package_id]
        )

    def resolve_package_path(
        self,
        package_id: str,
        *,
        validate: bool = True,
    ) -> Path:
        entry = self.get(package_id)
        path = self._resolved_path(entry.package_path)
        if validate:
            result = validate_personal_model_package(path)
            result.raise_for_error()
        return path

    def resolve_active_runtime(
        self,
        *,
        user_id: str,
        task: str,
        validate: bool = True,
    ) -> Path:
        entry = self.get_active(user_id=user_id, task=task)
        if not entry.runtime_ready:
            raise ValueError(
                f"Active package {entry.package_id} is not runtime-ready."
            )
        package_path = self._resolved_path(entry.package_path)
        loaded = load_personal_model_package(
            package_path,
            validate=validate,
        )
        runtime_path = loaded.runtime_path
        if runtime_path is None:
            raise ValueError(
                f"Package {entry.package_id} has no runtime path."
            )
        return runtime_path

    def archive(self, package_id: str) -> RegistryEntry:
        with _RegistryLock(self.lock_path):
            payload = self._read_payload()
            if package_id not in payload["entries"]:
                raise KeyError(
                    f"Package ID is not registered: {package_id}"
                )
            entries = dict(payload["entries"])
            entry = dict(entries[package_id])
            entry["status"] = "archived"
            entries[package_id] = entry

            active = dict(payload["active"])
            for key, active_id in list(active.items()):
                if active_id == package_id:
                    del active[key]

            payload["entries"] = entries
            payload["active"] = active
            self._write_payload(payload)
        return self.get(package_id)

    def remove(
        self,
        package_id: str,
        *,
        allow_active: bool = False,
    ) -> RegistryEntry:
        """Remove only the registry entry; model files are never deleted."""

        with _RegistryLock(self.lock_path):
            payload = self._read_payload()
            if package_id not in payload["entries"]:
                raise KeyError(
                    f"Package ID is not registered: {package_id}"
                )
            entry = RegistryEntry.from_dict(
                payload["entries"][package_id]
            )
            active = dict(payload["active"])
            active_keys = [
                key for key, value in active.items()
                if value == package_id
            ]
            if active_keys and not allow_active:
                raise ValueError(
                    f"Package {package_id} is active. Clear/archive it "
                    "before removal, or pass allow_active=True."
                )
            for key in active_keys:
                del active[key]

            entries = dict(payload["entries"])
            del entries[package_id]
            payload["entries"] = entries
            payload["active"] = active
            self._write_payload(payload)
        return entry

    def verify_all(self) -> dict[str, list[str]]:
        """Return package validation errors keyed by package ID."""

        failures: dict[str, list[str]] = {}
        for entry in self.list_entries():
            path = self._resolved_path(entry.package_path)
            result = validate_personal_model_package(path)
            if not result.valid:
                failures[entry.package_id] = list(result.errors)
        return failures

    def register_many(
        self,
        package_paths: Iterable[str | Path],
        *,
        status: RegistryStatus = "candidate",
        replace: bool = False,
    ) -> list[RegistryEntry]:
        return [
            self.register(
                path,
                status=status,
                replace=replace,
            )
            for path in package_paths
        ]
