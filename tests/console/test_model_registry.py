from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

from app.services import model_service
from app.services.model_service import ModelRegistry


def test_model_package_discovery_is_stable_and_privacy_safe(runtime_package: Path) -> None:
    registry = ModelRegistry(
        [runtime_package.parents[3]],
        runtime_verifier=lambda _path: True,
        live_verifier=lambda _path: True,
    )
    first = registry.list()
    second = registry.list()
    assert len(first) == 1
    assert first[0].id == second[0].id
    assert first[0].model_name == "50M"
    assert first[0].subject_id == "S01"
    assert first[0].runtime_verified is True
    assert first[0].live_verified is True
    serialized = json.dumps(first[0].model_dump())
    assert str(runtime_package) not in serialized
    assert ":\\" not in serialized


def test_invalid_model_package_is_ignored_with_safe_diagnostics(runtime_package: Path, caplog) -> None:
    invalid = runtime_package.parents[2] / "invalid" / "v1"
    invalid.mkdir(parents=True)
    (invalid / "package.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    registry = ModelRegistry([runtime_package.parents[3]], runtime_verifier=lambda _path: True)
    with caplog.at_level(logging.WARNING):
        assert len(registry.list()) == 1
    assert "skipped runtime package" in caplog.text
    assert "invalid/v1/package.yaml" in caplog.text
    assert str(runtime_package.parents[3]) not in caplog.text


def test_formal_live_verification_uses_loader_policy_and_cache(
    runtime_package: Path,
    monkeypatch,
) -> None:
    calls = {"loader": 0, "policy": 0}
    package = SimpleNamespace(window_sec=4.0, step_sec=0.5)

    def load_package(_path: Path, *, device: str, verify_hashes: bool):
        calls["loader"] += 1
        assert device == "cpu"
        assert verify_hashes is True
        return package

    def create_policy(value: object):
        calls["policy"] += 1
        assert value is package
        return SimpleNamespace()

    monkeypatch.setattr(model_service, "load_runtime_package", load_package)
    monkeypatch.setattr(model_service.RealtimeModelPolicyRegistry, "create", create_policy)
    registry = ModelRegistry([runtime_package.parents[3]])

    assert registry._verify_live(runtime_package) is True
    assert registry._verify_live(runtime_package) is True
    assert calls == {"loader": 1, "policy": 1}
