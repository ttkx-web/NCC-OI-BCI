from __future__ import annotations

import json
from pathlib import Path

import pytest

from bci_dayloop.personalization import (
    create_personal_model_package,
    load_personal_model_package,
    resolve_runtime_package_path,
    validate_personal_model_package,
)


RUNTIME_REQUIRED_FILES = (
    "model.yaml",
    "preprocessing.yaml",
    "classifier.pt",
    "label_map.json",
    "command_map.json",
    "base_model.json",
)
CLASS_NAMES = ("left_hand", "right_hand", "feet", "tongue")
INPUT_CONTRACT = {
    "window_seconds": 4.0,
    "target_sample_rate": 100.0,
    "num_tokens": 256,
    "model_n_time_patches": 10,
    "aggregation": "flatten",
}


def _write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_runtime_package(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    (path / "model.yaml").write_text(
        "name: 50m-linear\n"
        "window_seconds: 4.0\n"
        "step_sec: 0.5\n",
        encoding="utf-8",
    )
    (path / "preprocessing.yaml").write_text(
        "target_sample_rate: 100.0\n",
        encoding="utf-8",
    )
    (path / "classifier.pt").write_bytes(b"population-runtime-head")
    (path / "label_map.json").write_text(
        json.dumps({str(i): name for i, name in enumerate(CLASS_NAMES)}),
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


def _create_head_only_runtime_package(
    tmp_path: Path,
    *,
    output_name: str = "personal-package",
    classifier_payload: bytes = b"personal-head-v1",
):
    runtime = _write_runtime_package(tmp_path / f"{output_name}-runtime")
    classifier = _write_bytes(
        tmp_path / f"{output_name}-head.pt",
        classifier_payload,
    )
    backbone = _write_bytes(
        tmp_path / "model_deploy.pt",
        b"shared-50m-backbone",
    )
    output = tmp_path / output_name

    loaded = create_personal_model_package(
        output_dir=output,
        user_id="subject_01",
        task="motor_imagery_4class",
        adaptation_type="head_only",
        classifier_checkpoint=classifier,
        base_backbone_checkpoint=backbone,
        runtime_package_dir=runtime,
        class_names=CLASS_NAMES,
        input_contract=INPUT_CONTRACT,
        metrics={"personal_validation_bacc": 0.75},
        training_metadata={"trials_per_class": 20, "seed": 42},
        package_id=f"{output_name}-id",
        git_commit="deadbeef",
        notes=("pytest fixture",),
    )
    return loaded, output, classifier, backbone


def test_head_only_runtime_package_is_directly_loadable_and_valid(
    tmp_path: Path,
) -> None:
    loaded, output, classifier, backbone = _create_head_only_runtime_package(
        tmp_path
    )

    assert loaded.path == output.resolve()
    assert loaded.runtime_path == output.resolve()
    assert loaded.classifier_path == (output / "classifier.pt").resolve()
    assert loaded.classifier_path.read_bytes() == classifier.read_bytes()

    assert loaded.manifest.package_id == "personal-package-id"
    assert loaded.manifest.user_id == "subject_01"
    assert loaded.manifest.task == "motor_imagery_4class"
    assert loaded.manifest.model_scope == "personalized"
    assert loaded.manifest.adaptation_type == "head_only"
    assert loaded.manifest.runtime_ready is True
    assert loaded.manifest.personalized_backbone_path is None
    assert loaded.manifest.class_names == CLASS_NAMES
    assert loaded.manifest.input_contract == INPUT_CONTRACT
    assert Path(loaded.manifest.base_backbone_path) == backbone.resolve()

    for filename in RUNTIME_REQUIRED_FILES:
        assert (output / filename).is_file(), filename
    for filename in (
        "personalization.json",
        "metrics.json",
        "training.json",
    ):
        assert (output / filename).is_file(), filename

    base_model = json.loads(
        (output / "base_model.json").read_text(encoding="utf-8")
    )
    assert base_model["model_scope"] == "personalized"
    assert base_model["user_id"] == "subject_01"
    assert base_model["task"] == "motor_imagery_4class"
    assert base_model["adaptation_type"] == "head_only"
    assert base_model["classifier_path"] == "classifier.pt"
    assert base_model["base_backbone_path"] == str(backbone.resolve())

    validation = validate_personal_model_package(output)
    assert validation.valid
    assert validation.runtime_ready
    assert validation.errors == ()
    assert resolve_runtime_package_path(output) == output.resolve()

    reloaded = load_personal_model_package(output)
    assert reloaded.manifest.to_dict() == loaded.manifest.to_dict()


def test_artifact_only_package_is_not_runtime_ready(tmp_path: Path) -> None:
    classifier = _write_bytes(tmp_path / "head.pt", b"personal-head")
    backbone = _write_bytes(tmp_path / "backbone.pt", b"shared-backbone")
    output = tmp_path / "artifact-only"

    loaded = create_personal_model_package(
        output_dir=output,
        user_id="subject_01",
        task="motor_imagery_4class",
        adaptation_type="head_only",
        classifier_checkpoint=classifier,
        base_backbone_checkpoint=backbone,
        class_names=CLASS_NAMES,
        package_id="artifact-only-id",
    )

    assert loaded.manifest.runtime_ready is False
    assert loaded.runtime_path is None
    assert loaded.classifier_path == (
        output / "artifacts" / "classifier.pt"
    ).resolve()
    assert loaded.classifier_path.read_bytes() == classifier.read_bytes()

    validation = validate_personal_model_package(output)
    assert validation.valid
    assert not validation.runtime_ready

    with pytest.raises(ValueError, match="not runtime-ready"):
        resolve_runtime_package_path(output)


def test_creation_refuses_overwrite_unless_explicitly_enabled(
    tmp_path: Path,
) -> None:
    runtime = _write_runtime_package(tmp_path / "runtime")
    first_classifier = _write_bytes(tmp_path / "first.pt", b"first-head")
    second_classifier = _write_bytes(tmp_path / "second.pt", b"second-head")
    backbone = _write_bytes(tmp_path / "backbone.pt", b"backbone")
    output = tmp_path / "versioned-package"

    common = {
        "output_dir": output,
        "user_id": "subject_01",
        "task": "motor_imagery_4class",
        "adaptation_type": "head_only",
        "base_backbone_checkpoint": backbone,
        "runtime_package_dir": runtime,
        "class_names": CLASS_NAMES,
        "package_id": "versioned-package-id",
    }

    create_personal_model_package(
        classifier_checkpoint=first_classifier,
        **common,
    )

    with pytest.raises(FileExistsError, match="Package already exists"):
        create_personal_model_package(
            classifier_checkpoint=second_classifier,
            **common,
        )

    replaced = create_personal_model_package(
        classifier_checkpoint=second_classifier,
        overwrite=True,
        **common,
    )

    assert replaced.classifier_path.read_bytes() == b"second-head"
    assert not Path(f"{output}.backup").exists()


def test_partial_finetune_requires_and_packages_personalized_backbone(
    tmp_path: Path,
) -> None:
    runtime = _write_runtime_package(tmp_path / "runtime")
    classifier = _write_bytes(tmp_path / "head.pt", b"personal-head")
    base_backbone = _write_bytes(
        tmp_path / "base-backbone.pt",
        b"base-backbone",
    )
    personalized_backbone = _write_bytes(
        tmp_path / "personal-backbone.pt",
        b"personal-backbone",
    )
    output = tmp_path / "partial-package"

    loaded = create_personal_model_package(
        output_dir=output,
        user_id="subject_01",
        task="motor_imagery_4class",
        adaptation_type="partial_finetune",
        classifier_checkpoint=classifier,
        base_backbone_checkpoint=base_backbone,
        personalized_backbone_checkpoint=personalized_backbone,
        runtime_package_dir=runtime,
        class_names=CLASS_NAMES,
        package_id="partial-package-id",
    )

    assert loaded.personalized_backbone_path == (
        output / "backbone.pt"
    ).resolve()
    assert loaded.personalized_backbone_path.read_bytes() == (
        personalized_backbone.read_bytes()
    )

    base_model = json.loads(
        (output / "base_model.json").read_text(encoding="utf-8")
    )
    assert base_model["checkpoint_path"] == "backbone.pt"
    assert base_model["checkpoint_path_absolute"] == str(
        (output / "backbone.pt").resolve()
    )
    assert base_model["personalized_backbone"] is True
    assert validate_personal_model_package(output).valid


@pytest.mark.parametrize("adaptation_type", ["partial_finetune", "full_finetune"])
def test_backbone_finetune_packages_require_personalized_checkpoint(
    tmp_path: Path,
    adaptation_type: str,
) -> None:
    classifier = _write_bytes(tmp_path / "head.pt", b"head")
    backbone = _write_bytes(tmp_path / "backbone.pt", b"backbone")

    with pytest.raises(ValueError, match="requires personalized_backbone"):
        create_personal_model_package(
            output_dir=tmp_path / adaptation_type,
            user_id="subject_01",
            task="motor_imagery_4class",
            adaptation_type=adaptation_type,
            classifier_checkpoint=classifier,
            base_backbone_checkpoint=backbone,
        )


def test_head_only_package_rejects_personalized_backbone(tmp_path: Path) -> None:
    classifier = _write_bytes(tmp_path / "head.pt", b"head")
    base_backbone = _write_bytes(tmp_path / "base.pt", b"base")
    personal_backbone = _write_bytes(tmp_path / "personal.pt", b"personal")

    with pytest.raises(ValueError, match="must not provide"):
        create_personal_model_package(
            output_dir=tmp_path / "invalid-head-only",
            user_id="subject_01",
            task="motor_imagery_4class",
            adaptation_type="head_only",
            classifier_checkpoint=classifier,
            base_backbone_checkpoint=base_backbone,
            personalized_backbone_checkpoint=personal_backbone,
        )


def test_validation_detects_classifier_tampering(tmp_path: Path) -> None:
    loaded, output, _, _ = _create_head_only_runtime_package(
        tmp_path,
        output_name="tamper-test",
    )
    loaded.classifier_path.write_bytes(b"tampered-checkpoint")

    validation = validate_personal_model_package(output)

    assert not validation.valid
    assert any("Classifier SHA-256 mismatch" in item for item in validation.errors)
    with pytest.raises(ValueError, match="Classifier SHA-256 mismatch"):
        load_personal_model_package(output)


def test_runtime_package_must_contain_all_required_files(tmp_path: Path) -> None:
    runtime = _write_runtime_package(tmp_path / "incomplete-runtime")
    (runtime / "command_map.json").unlink()
    classifier = _write_bytes(tmp_path / "head.pt", b"head")
    backbone = _write_bytes(tmp_path / "backbone.pt", b"backbone")

    with pytest.raises(FileNotFoundError, match="command_map.json"):
        create_personal_model_package(
            output_dir=tmp_path / "output",
            user_id="subject_01",
            task="motor_imagery_4class",
            adaptation_type="head_only",
            classifier_checkpoint=classifier,
            base_backbone_checkpoint=backbone,
            runtime_package_dir=runtime,
        )
