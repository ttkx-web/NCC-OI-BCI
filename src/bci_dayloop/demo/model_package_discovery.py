"""Read-only discovery of compatible Runtime Model Packages for the demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bci_dayloop.demo.motor_decoder import _canonical_motor_label
from bci_dayloop.utils.config import load_yaml


@dataclass(frozen=True, slots=True)
class MotorPackageOption:
    path: str
    label: str
    model_name: str
    model_type: str
    window_sec: float


def _is_motor_imagery_package(payload: dict[str, Any]) -> bool:
    model = payload.get("model")
    if not isinstance(model, dict) or str(model.get("task", "")).lower() != "motor_imagery":
        return False
    if int(model.get("num_classes", 0)) != 4:
        return False
    class_names = model.get("class_names")
    if not isinstance(class_names, list):
        return False
    try:
        return len(class_names) == 4 and set(_canonical_motor_label(str(name)) for name in class_names) == {
            "left_hand",
            "right_hand",
            "feet",
            "tongue",
        }
    except ValueError:
        return False


def discover_motor_intent_packages(root: str | Path) -> list[MotorPackageOption]:
    """Discover schema-v2, four-class MI packages under existing project roots."""
    repository_root = Path(root).resolve()
    candidates: list[MotorPackageOption] = []
    for relative_root in ("model_packages", "runs", "checkpoints"):
        search_root = repository_root / relative_root
        if not search_root.is_dir():
            continue
        for package_yaml in search_root.rglob("package.yaml"):
            try:
                payload = load_yaml(package_yaml)
                if int(payload.get("schema_version", -1)) != 2 or not _is_motor_imagery_package(payload):
                    continue
                model = dict(payload["model"])
                contract = dict(payload.get("input_contract", {}))
                family = {"model_50m": "50M", "labram": "LaBraM", "cbramod": "CBraMod"}.get(
                    str(model.get("type", "")),
                    str(model.get("name", package_yaml.parent.name)),
                )
                dataset = str(model.get("dataset", "BNCI")).replace("BNCI2014_001", "BNCI")
                window_sec = float(contract["window_sec"])
                relative = package_yaml.parent.relative_to(repository_root)
                origin = "/".join(relative.parts[:2])
                candidates.append(
                    MotorPackageOption(
                        path=str(package_yaml.parent),
                        label=f"{family} · {dataset} · {window_sec:g}s · {origin}",
                        model_name=str(model.get("name", family)),
                        model_type=str(model.get("type", "")),
                        window_sec=window_sec,
                    )
                )
            except Exception:
                # Discovery must not make the UI fail because one package is incomplete.
                continue
    return sorted(candidates, key=lambda option: (abs(option.window_sec - 2.0), option.label, option.path))
