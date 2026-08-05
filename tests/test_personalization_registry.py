from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bci_dayloop.personalization import (
    PersonalModelRegistry,
    create_personal_model_package,
)


CLASS_NAMES = ("left_hand", "right_hand", "feet", "tongue")
INPUT_CONTRACT = {
    "window_seconds": 4.0,
    "target_sample_rate": 100.0,
    "num_tokens": 256,
    "model_n_time_patches": 10,
    "aggregation": "flatten",
}


def _write_file(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_runtime_package(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    (path / "model.yaml").write_text(
        "name: 50m-linear\n"
        "num_classes: 4\n"
        "window_seconds: 4.0\n"
        "step_sec: 0.5\n"
        "target_sample_rate: 100.0\n"
        "aggregation: flatten\n"
        "model_n_time_patches: 10\n"
        "class_names:\n"
        "  - left_hand\n"
        "  - right_hand\n"
        "  - feet\n"
        "  - tongue\n",
        encoding="utf-8",
    )
    (path / "preprocessing.yaml").write_text(
        "target_sample_rate: 100.0\n"
        "strict_window_duration: true\n",
        encoding="utf-8",
    )
    (path / "classifier.pt").write_bytes(b"runtime-classifier")
    (path / "label_map.json").write_text(
        json.dumps({str(i): name for i, name in enumerate(CLASS_NAMES)})
        + "\n",
        encoding="utf-8",
    )
    (path / "command_map.json").write_text("{}\n", encoding="utf-8")
    (path / "base_model.json").write_text(
        json.dumps(
            {
                "checkpoint_path": "shared_backbone.pt",
                "is_test_head": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _create_package(
    tmp_path: Path,
    *,
    package_id: str,
    user_id: str = "subject_01",
    task: str = "motor_imagery_4class",
    runtime_ready: bool = True,
    metric: float = 0.75,
) -> Path:
    source_root = tmp_path / "sources" / package_id
    classifier = _write_file(
        source_root / "personal_head.pt",
        f"classifier:{package_id}".encode(),
    )
    backbone = _write_file(
        source_root / "model_deploy.pt",
        b"shared-backbone",
    )
    runtime = (
        _write_runtime_package(source_root / "runtime_package")
        if runtime_ready
        else None
    )
    output = tmp_path / "packages" / package_id

    create_personal_model_package(
        output_dir=output,
        user_id=user_id,
        task=task,
        adaptation_type="head_only",
        classifier_checkpoint=classifier,
        base_backbone_checkpoint=backbone,
        runtime_package_dir=runtime,
        class_names=CLASS_NAMES,
        input_contract=INPUT_CONTRACT,
        metrics={"personal_validation_bacc": metric},
        training_metadata={"trials_per_class": 20, "seed": 42},
        package_id=package_id,
        notes=(f"package {package_id}",),
    )
    return output


def test_registry_creation_writes_empty_versioned_payload(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registries" / "stage1_personal_models.json"

    registry = PersonalModelRegistry(registry_path)

    assert registry.path == registry_path.resolve()
    assert registry_path.is_file()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert payload["format_version"] == 1
    assert payload["entries"] == {}
    assert payload["active"] == {}
    assert registry.list_entries() == []


def test_register_candidate_and_resolve_package_path(tmp_path: Path) -> None:
    package = _create_package(tmp_path, package_id="candidate-v1")
    registry = PersonalModelRegistry(
        tmp_path / "registries" / "stage1_personal_models.json"
    )

    entry = registry.register(package)

    assert entry.package_id == "candidate-v1"
    assert entry.user_id == "subject_01"
    assert entry.task == "motor_imagery_4class"
    assert entry.status == "candidate"
    assert entry.runtime_ready is True
    assert entry.metrics["personal_validation_bacc"] == pytest.approx(0.75)
    assert registry.get(entry.package_id) == entry
    assert registry.resolve_package_path(entry.package_id) == package.resolve()

    # The registry stores a portable relative path when possible.
    assert not Path(entry.package_path).is_absolute()

    expected_stored_path = Path(
        os.path.relpath(
            package.resolve(),
            start=registry.path.parent,
        )
    )

    assert Path(entry.package_path) == expected_stored_path


def test_register_with_set_active_resolves_runtime_root(
    tmp_path: Path,
) -> None:
    package = _create_package(tmp_path, package_id="active-v1")
    registry = PersonalModelRegistry(
        tmp_path / "registries" / "stage1_personal_models.json"
    )

    entry = registry.register(package, set_active=True)

    assert entry.status == "active"
    assert registry.get_active(
        user_id="subject_01",
        task="motor_imagery_4class",
    ).package_id == "active-v1"
    assert registry.resolve_active_runtime(
        user_id="subject_01",
        task="motor_imagery_4class",
    ) == package.resolve()


def test_activating_new_version_demotes_previous_version(
    tmp_path: Path,
) -> None:
    first = _create_package(tmp_path, package_id="personal-v1", metric=0.70)
    second = _create_package(tmp_path, package_id="personal-v2", metric=0.80)
    registry = PersonalModelRegistry(
        tmp_path / "registries" / "stage1_personal_models.json"
    )

    registry.register(first, set_active=True)
    registry.register(second, set_active=True)

    assert registry.get("personal-v1").status == "candidate"
    assert registry.get("personal-v2").status == "active"
    assert registry.get_active(
        user_id="subject_01",
        task="motor_imagery_4class",
    ).package_id == "personal-v2"


def test_artifact_only_package_cannot_be_activated(tmp_path: Path) -> None:
    package = _create_package(
        tmp_path,
        package_id="artifact-only",
        runtime_ready=False,
    )
    registry = PersonalModelRegistry(
        tmp_path / "registries" / "stage1_personal_models.json"
    )

    entry = registry.register(package)
    assert entry.runtime_ready is False

    with pytest.raises(ValueError, match="not runtime-ready"):
        registry.set_active(
            user_id="subject_01",
            task="motor_imagery_4class",
            package_id=entry.package_id,
        )


def test_duplicate_package_id_requires_replace(tmp_path: Path) -> None:
    package = _create_package(tmp_path, package_id="duplicate-v1")
    registry = PersonalModelRegistry(
        tmp_path / "registries" / "stage1_personal_models.json"
    )
    registry.register(package)

    with pytest.raises(KeyError, match="already registered"):
        registry.register(package)

    replaced = registry.register(package, replace=True)
    assert replaced.package_id == "duplicate-v1"


def test_list_entries_filters_by_user_task_status_and_runtime(
    tmp_path: Path,
) -> None:
    package_a = _create_package(
        tmp_path,
        package_id="s01-runtime",
        user_id="subject_01",
        runtime_ready=True,
    )
    package_b = _create_package(
        tmp_path,
        package_id="s01-artifact",
        user_id="subject_01",
        runtime_ready=False,
    )
    package_c = _create_package(
        tmp_path,
        package_id="s02-runtime",
        user_id="subject_02",
        runtime_ready=True,
    )
    registry = PersonalModelRegistry(
        tmp_path / "registries" / "stage1_personal_models.json"
    )
    registry.register_many((package_a, package_b, package_c))
    registry.set_active(
        user_id="subject_01",
        task="motor_imagery_4class",
        package_id="s01-runtime",
    )

    assert {
        entry.package_id
        for entry in registry.list_entries(user_id="subject_01")
    } == {"s01-artifact", "s01-runtime"}
    assert [
        entry.package_id
        for entry in registry.list_entries(status="active")
    ] == ["s01-runtime"]
    assert {
        entry.package_id
        for entry in registry.list_entries(runtime_ready=False)
    } == {"s01-artifact"}


def test_clear_and_archive_remove_active_mapping(tmp_path: Path) -> None:
    package = _create_package(tmp_path, package_id="archive-v1")
    registry = PersonalModelRegistry(
        tmp_path / "registries" / "stage1_personal_models.json"
    )
    registry.register(package, set_active=True)

    registry.clear_active(
        user_id="subject_01",
        task="motor_imagery_4class",
    )
    assert registry.get("archive-v1").status == "candidate"
    with pytest.raises(KeyError, match="No active package"):
        registry.get_active(
            user_id="subject_01",
            task="motor_imagery_4class",
        )

    registry.set_active(
        user_id="subject_01",
        task="motor_imagery_4class",
        package_id="archive-v1",
    )
    archived = registry.archive("archive-v1")
    assert archived.status == "archived"
    with pytest.raises(KeyError, match="No active package"):
        registry.get_active(
            user_id="subject_01",
            task="motor_imagery_4class",
        )


def test_remove_never_deletes_package_files(tmp_path: Path) -> None:
    package = _create_package(tmp_path, package_id="remove-v1")
    registry = PersonalModelRegistry(
        tmp_path / "registries" / "stage1_personal_models.json"
    )
    registry.register(package, set_active=True)

    with pytest.raises(ValueError, match="is active"):
        registry.remove("remove-v1")

    removed = registry.remove("remove-v1", allow_active=True)
    assert removed.package_id == "remove-v1"
    assert package.is_dir()
    with pytest.raises(KeyError, match="not registered"):
        registry.get("remove-v1")


def test_verify_all_reports_hash_tampering(tmp_path: Path) -> None:
    healthy = _create_package(tmp_path, package_id="healthy-v1")
    tampered = _create_package(tmp_path, package_id="tampered-v1")
    registry = PersonalModelRegistry(
        tmp_path / "registries" / "stage1_personal_models.json"
    )
    registry.register_many((healthy, tampered))

    (tampered / "classifier.pt").write_bytes(b"modified-after-registration")

    failures = registry.verify_all()

    assert "healthy-v1" not in failures
    assert "tampered-v1" in failures
    assert any(
        "Classifier SHA-256 mismatch" in message
        for message in failures["tampered-v1"]
    )
