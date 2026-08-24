"""Small public utilities shared by runtime-package implementations."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def safe_torch_load(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path}: checkpoint must be a mapping.")
    return payload

def required_mapping(payload: Mapping[str, Any], key: str, *, source: Path) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{source} field {key!r} must be a mapping.")
    return dict(value)

def resolve_package_file(package_path: Path, value: str, *, logical_name: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{logical_name} must use a package-relative path, got {value!r}.")
    resolved = (package_path / relative).resolve()
    try:
        resolved.relative_to(package_path)
    except ValueError as error:
        raise ValueError(f"{logical_name} escapes package directory: {value!r}.") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{logical_name} was not found: {resolved}")
    return resolved

def verify_sha256(*, path: Path, expected: str | None, logical_name: str) -> None:
    if expected:
        actual = sha256_file(path)
        if actual.lower() != str(expected).lower():
            raise ValueError(f"{logical_name} SHA256 mismatch: expected={expected}, actual={actual}.")
