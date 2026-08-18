"""Validated device-specific 2D anchor configs for cortical demo overlays."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


MONTAGE_CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "cortical_montages"
_VALID_HEMISPHERES = {"left", "right"}


def canonical_channel_name(name: str) -> str:
    normalized = name.upper().strip().replace(" ", "")
    return normalized.removeprefix("EEG-").removeprefix("EEG")


@dataclass(frozen=True, slots=True)
class CorticalAnchor:
    hemisphere: str
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class CorticalMontage:
    name: str
    device_name: str
    description: str
    channels: dict[str, tuple[CorticalAnchor, ...]]


def _anchor(payload: object, *, channel: str) -> CorticalAnchor:
    if not isinstance(payload, dict):
        raise ValueError(f"{channel}: every anchor must be an object")
    hemisphere = payload.get("hemisphere")
    if hemisphere not in _VALID_HEMISPHERES:
        raise ValueError(f"{channel}: hemisphere must be one of {sorted(_VALID_HEMISPHERES)}")
    try:
        x, y = float(payload["x"]), float(payload["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{channel}: anchor must include numeric x/y") from exc
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        raise ValueError(f"{channel}: anchor x/y must be in [0, 1]")
    return CorticalAnchor(hemisphere=hemisphere, x=x, y=y)


def validate_cortical_montage(payload: object, *, expected_name: str | None = None) -> CorticalMontage:
    """Validate JSON payload once at load time, never in the decode loop."""
    if not isinstance(payload, dict):
        raise ValueError("cortical montage config must be a JSON object")
    name = payload.get("montage_name")
    device_name = payload.get("device_name")
    channels_payload = payload.get("channels")
    if not isinstance(name, str) or not name:
        raise ValueError("montage_name must be a non-empty string")
    if expected_name is not None and name != expected_name:
        raise ValueError(f"montage_name {name!r} does not match requested config {expected_name!r}")
    if not isinstance(device_name, str) or not device_name:
        raise ValueError("device_name must be a non-empty string")
    if not isinstance(channels_payload, dict) or not channels_payload:
        raise ValueError("channels must contain at least one channel")
    channels: dict[str, tuple[CorticalAnchor, ...]] = {}
    for raw_name, channel_payload in channels_payload.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("channel names must be non-empty strings")
        name_key = canonical_channel_name(raw_name)
        if name_key in channels:
            raise ValueError(f"duplicate channel name after canonicalization: {raw_name!r}")
        if not isinstance(channel_payload, dict):
            raise ValueError(f"{raw_name}: channel entry must be an object")
        anchors_payload = channel_payload.get("anchors")
        if not isinstance(anchors_payload, list) or not anchors_payload:
            raise ValueError(f"{raw_name}: anchors must be a non-empty list")
        anchors = tuple(_anchor(anchor_payload, channel=raw_name) for anchor_payload in anchors_payload)
        channels[name_key] = anchors
    return CorticalMontage(
        name=name,
        device_name=device_name,
        description=str(payload.get("description", "")),
        channels=channels,
    )


@lru_cache(maxsize=16)
def load_cortical_montage(name: str) -> CorticalMontage:
    """Load a registry config by name; filenames are not user-provided paths."""
    if not name or Path(name).name != name or name.endswith(".json"):
        raise ValueError("montage name must be a simple registry name without '.json'")
    path = MONTAGE_CONFIG_DIR / f"{name}.json"
    if not path.exists():
        available = ", ".join(list_cortical_montages()) or "(none)"
        raise ValueError(f"unknown cortical montage {name!r}; available: {available}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in cortical montage {path}") from exc
    return validate_cortical_montage(payload, expected_name=name)


def list_cortical_montages() -> list[str]:
    if not MONTAGE_CONFIG_DIR.exists():
        return []
    return sorted(path.stem for path in MONTAGE_CONFIG_DIR.glob("*.json") if path.name != "cortical_montage_template.json")


__all__ = [
    "CorticalAnchor",
    "CorticalMontage",
    "MONTAGE_CONFIG_DIR",
    "canonical_channel_name",
    "list_cortical_montages",
    "load_cortical_montage",
    "validate_cortical_montage",
]
