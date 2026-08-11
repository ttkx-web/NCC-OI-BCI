from __future__ import annotations

import json
from pathlib import Path

from app.services.model_service import ModelRegistry


def test_model_package_discovery_is_stable_and_privacy_safe(runtime_package: Path) -> None:
    registry = ModelRegistry([runtime_package.parents[3]])
    first = registry.list()
    second = registry.list()
    assert len(first) == 1
    assert first[0].id == second[0].id
    assert first[0].model_name == "50M"
    assert first[0].subject_id == "S01"
    assert first[0].runtime_verified is True
    serialized = json.dumps(first[0].model_dump())
    assert str(runtime_package) not in serialized
    assert ":\\" not in serialized


def test_invalid_model_package_is_ignored(runtime_package: Path) -> None:
    invalid = runtime_package.parents[2] / "invalid" / "v1"
    invalid.mkdir(parents=True)
    (invalid / "package.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    registry = ModelRegistry([runtime_package.parents[3]])
    assert len(registry.list()) == 1
