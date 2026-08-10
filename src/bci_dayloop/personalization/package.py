from __future__ import annotations

"""Personal-model package creation and validation.

A package created from an existing 50M Runtime Model Package remains directly
loadable by the current CLI/Streamlit runtime: the standard runtime files stay
at the package root, and personalization-specific files are added alongside
them.

Head-only package layout::

    package/
    ├── model.yaml
    ├── preprocessing.yaml
    ├── classifier.pt
    ├── label_map.json
    ├── command_map.json
    ├── base_model.json
    ├── personalization.json
    ├── training.json
    └── metrics.json

When no runtime package is supplied, the module creates a training-artifact
package under ``artifacts/``.  Such a package can be registered, but is marked
``runtime_ready=False`` until a runtime package is exported.
"""

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence


AdaptationType = Literal[
    "head_only",
    "partial_finetune",
    "full_finetune",
    "lora",
    "rest_tuning",
]

RUNTIME_REQUIRED_FILES = (
    "model.yaml",
    "preprocessing.yaml",
    "classifier.pt",
    "label_map.json",
    "command_map.json",
    "base_model.json",
)

PERSONALIZATION_MANIFEST = "personalization.json"


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_file(
    path: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"File was not found: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(dict(payload)),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return dict(payload)


def _safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    value = value.strip("._-")
    if not value:
        raise ValueError("Package path component cannot be empty.")
    return value


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Source file was not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{destination}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _copy_runtime_package(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise NotADirectoryError(
            f"Runtime package directory was not found: {source}"
        )
    missing = [
        name for name in RUNTIME_REQUIRED_FILES
        if not (source / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Runtime package {source} is missing {missing}."
        )

    for item in source.iterdir():
        target = destination / item.name
        if item.is_symlink():
            raise ValueError(
                f"Runtime package must not contain symlinks: {item}"
            )
        if item.is_dir():
            shutil.copytree(item, target)
        elif item.is_file():
            shutil.copy2(item, target)


@dataclass(frozen=True, slots=True)
class PersonalModelManifest:
    format_version: int
    package_id: str
    user_id: str
    task: str
    model_scope: str
    adaptation_type: str
    created_at: str
    runtime_ready: bool
    runtime_package_path: str | None
    classifier_path: str
    classifier_sha256: str
    base_backbone_path: str
    base_backbone_sha256: str
    personalized_backbone_path: str | None
    personalized_backbone_sha256: str | None
    class_names: tuple[str, ...]
    input_contract: dict[str, Any]
    metrics_path: str | None
    training_path: str | None
    git_commit: str | None
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": int(self.format_version),
            "package_id": self.package_id,
            "user_id": self.user_id,
            "task": self.task,
            "model_scope": self.model_scope,
            "adaptation_type": self.adaptation_type,
            "created_at": self.created_at,
            "runtime_ready": bool(self.runtime_ready),
            "runtime_package_path": self.runtime_package_path,
            "classifier": {
                "path": self.classifier_path,
                "sha256": self.classifier_sha256,
            },
            "base_backbone": {
                "path": self.base_backbone_path,
                "sha256": self.base_backbone_sha256,
            },
            "personalized_backbone": (
                {
                    "path": self.personalized_backbone_path,
                    "sha256": self.personalized_backbone_sha256,
                }
                if self.personalized_backbone_path is not None
                else None
            ),
            "class_names": list(self.class_names),
            "input_contract": _json_safe(self.input_contract),
            "metrics_path": self.metrics_path,
            "training_path": self.training_path,
            "git_commit": self.git_commit,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "PersonalModelManifest":
        classifier = payload.get("classifier", {})
        base_backbone = payload.get("base_backbone", {})
        personalized = payload.get("personalized_backbone")
        if not isinstance(classifier, Mapping):
            raise TypeError("manifest.classifier must be a mapping.")
        if not isinstance(base_backbone, Mapping):
            raise TypeError("manifest.base_backbone must be a mapping.")
        if personalized is not None and not isinstance(
            personalized,
            Mapping,
        ):
            raise TypeError(
                "manifest.personalized_backbone must be a mapping or null."
            )
        return cls(
            format_version=int(payload.get("format_version", 1)),
            package_id=str(payload["package_id"]),
            user_id=str(payload["user_id"]),
            task=str(payload["task"]),
            model_scope=str(payload.get("model_scope", "personalized")),
            adaptation_type=str(payload["adaptation_type"]),
            created_at=str(payload["created_at"]),
            runtime_ready=bool(payload.get("runtime_ready", False)),
            runtime_package_path=(
                str(payload["runtime_package_path"])
                if payload.get("runtime_package_path") is not None
                else None
            ),
            classifier_path=str(classifier["path"]),
            classifier_sha256=str(classifier["sha256"]),
            base_backbone_path=str(base_backbone["path"]),
            base_backbone_sha256=str(base_backbone["sha256"]),
            personalized_backbone_path=(
                str(personalized["path"])
                if personalized is not None
                else None
            ),
            personalized_backbone_sha256=(
                str(personalized["sha256"])
                if personalized is not None
                else None
            ),
            class_names=tuple(
                str(value)
                for value in payload.get("class_names", [])
            ),
            input_contract=dict(payload.get("input_contract", {})),
            metrics_path=(
                str(payload["metrics_path"])
                if payload.get("metrics_path") is not None
                else None
            ),
            training_path=(
                str(payload["training_path"])
                if payload.get("training_path") is not None
                else None
            ),
            git_commit=(
                str(payload["git_commit"])
                if payload.get("git_commit") is not None
                else None
            ),
            notes=tuple(
                str(value) for value in payload.get("notes", [])
            ),
        )


@dataclass(frozen=True, slots=True)
class LoadedPersonalModelPackage:
    path: Path
    manifest: PersonalModelManifest

    @property
    def classifier_path(self) -> Path:
        return (self.path / self.manifest.classifier_path).resolve()

    @property
    def personalized_backbone_path(self) -> Path | None:
        value = self.manifest.personalized_backbone_path
        return (self.path / value).resolve() if value else None

    @property
    def runtime_path(self) -> Path | None:
        if not self.manifest.runtime_ready:
            return None
        value = self.manifest.runtime_package_path or "."
        return (self.path / value).resolve()


@dataclass(frozen=True, slots=True)
class PackageValidationResult:
    valid: bool
    package_path: Path
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    runtime_ready: bool

    def raise_for_error(self) -> None:
        if not self.valid:
            detail = "\n".join(f"  - {item}" for item in self.errors)
            raise ValueError(
                f"Invalid personal model package {self.package_path}:\n"
                f"{detail}"
            )


def _install_directory(
    temporary: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    backup = Path(f"{destination}.backup")
    if backup.exists():
        shutil.rmtree(backup)

    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Package already exists: {destination}. "
                "Pass overwrite=True to replace it."
            )
        destination.replace(backup)

    try:
        temporary.replace(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def create_personal_model_package(
    *,
    output_dir: str | Path,
    user_id: str,
    task: str,
    adaptation_type: AdaptationType,
    classifier_checkpoint: str | Path,
    base_backbone_checkpoint: str | Path,
    runtime_package_dir: str | Path | None = None,
    personalized_backbone_checkpoint: str | Path | None = None,
    class_names: Sequence[str] = (),
    input_contract: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    training_metadata: Mapping[str, Any] | None = None,
    package_id: str | None = None,
    git_commit: str | None = None,
    notes: Sequence[str] = (),
    overwrite: bool = False,
) -> LoadedPersonalModelPackage:
    """Create a personal-model package atomically.

    ``output_dir`` is the exact package directory.  When
    ``runtime_package_dir`` is provided, its standard runtime files are copied
    to the package root, so the returned path can be selected directly by the
    current CLI/Streamlit Model Package selector.
    """

    destination = Path(output_dir).expanduser().resolve()
    classifier_source = Path(
        classifier_checkpoint
    ).expanduser().resolve()
    base_backbone_source = Path(
        base_backbone_checkpoint
    ).expanduser().resolve()
    runtime_source = (
        Path(runtime_package_dir).expanduser().resolve()
        if runtime_package_dir is not None
        else None
    )
    personalized_source = (
        Path(personalized_backbone_checkpoint).expanduser().resolve()
        if personalized_backbone_checkpoint is not None
        else None
    )

    if not classifier_source.is_file():
        raise FileNotFoundError(
            f"Classifier checkpoint was not found: {classifier_source}"
        )
    if not base_backbone_source.is_file():
        raise FileNotFoundError(
            f"Base backbone checkpoint was not found: "
            f"{base_backbone_source}"
        )
    if personalized_source is not None and not personalized_source.is_file():
        raise FileNotFoundError(
            f"Personalized backbone checkpoint was not found: "
            f"{personalized_source}"
        )

    if adaptation_type == "head_only" and personalized_source is not None:
        raise ValueError(
            "head_only packages must not provide a personalized backbone."
        )
    if adaptation_type in {
        "partial_finetune",
        "full_finetune",
    } and personalized_source is None:
        raise ValueError(
            f"{adaptation_type} requires "
            "personalized_backbone_checkpoint."
        )

    created_at = utc_now_iso()
    if package_id is None:
        stamp = created_at.replace("-", "").replace(":", "")
        stamp = stamp.replace("T", "_").replace("Z", "")
        package_id = "__".join(
            (
                _safe_component(user_id),
                _safe_component(task),
                _safe_component(adaptation_type),
                stamp,
            )
        )
    else:
        package_id = _safe_component(package_id)

    temporary = destination.parent / (
        f".{destination.name}.tmp-{os.getpid()}"
    )
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=False)

    try:
        runtime_ready = runtime_source is not None
        if runtime_source is not None:
            _copy_runtime_package(runtime_source, temporary)
            classifier_relative = Path("classifier.pt")
            _copy_file(
                classifier_source,
                temporary / classifier_relative,
            )
        else:
            classifier_relative = Path("artifacts/classifier.pt")
            _copy_file(
                classifier_source,
                temporary / classifier_relative,
            )

        personalized_relative: Path | None = None
        if personalized_source is not None:
            personalized_relative = (
                Path("backbone.pt")
                if runtime_ready
                else Path("artifacts/backbone.pt")
            )
            _copy_file(
                personalized_source,
                temporary / personalized_relative,
            )

        metrics_relative: Path | None = None
        if metrics is not None:
            metrics_relative = Path("metrics.json")
            _atomic_write_json(
                temporary / metrics_relative,
                dict(metrics),
            )

        training_relative: Path | None = None
        if training_metadata is not None:
            training_relative = Path("training.json")
            _atomic_write_json(
                temporary / training_relative,
                dict(training_metadata),
            )

        base_sha = sha256_file(base_backbone_source)
        classifier_sha = sha256_file(
            temporary / classifier_relative
        )
        personalized_sha = (
            sha256_file(temporary / personalized_relative)
            if personalized_relative is not None
            else None
        )

        # Add personalization fields without changing the current runtime
        # loader's required keys.
        if runtime_ready:
            base_model_path = temporary / "base_model.json"
            base_model = _read_json(base_model_path)
            base_model.update(
                {
                    "model_scope": "personalized",
                    "user_id": str(user_id),
                    "task": str(task),
                    "adaptation_type": adaptation_type,
                    "classifier_path": str(classifier_relative),
                    "classifier_sha256": classifier_sha,
                    "base_backbone_path": str(base_backbone_source),
                    "base_backbone_sha256": base_sha,
                }
            )
            if personalized_relative is not None:
                # Current Model50MAdapter.from_package accepts a package-local
                # checkpoint path.  Keep an absolute value too for the current
                # loader, but the relative checkpoint_path is the portable
                # source of truth.
                base_model.update(
                    {
                        "checkpoint_path": str(personalized_relative),
                        "checkpoint_path_absolute": str(
                            (destination / personalized_relative).resolve()
                        ),
                        "checkpoint_sha256": personalized_sha,
                        "personalized_backbone": True,
                    }
                )
            _atomic_write_json(base_model_path, base_model)

        manifest = PersonalModelManifest(
            format_version=1,
            package_id=package_id,
            user_id=str(user_id),
            task=str(task),
            model_scope="personalized",
            adaptation_type=adaptation_type,
            created_at=created_at,
            runtime_ready=runtime_ready,
            runtime_package_path="." if runtime_ready else None,
            classifier_path=str(classifier_relative),
            classifier_sha256=classifier_sha,
            base_backbone_path=str(base_backbone_source),
            base_backbone_sha256=base_sha,
            personalized_backbone_path=(
                str(personalized_relative)
                if personalized_relative is not None
                else None
            ),
            personalized_backbone_sha256=personalized_sha,
            class_names=tuple(str(name) for name in class_names),
            input_contract=dict(input_contract or {}),
            metrics_path=(
                str(metrics_relative)
                if metrics_relative is not None
                else None
            ),
            training_path=(
                str(training_relative)
                if training_relative is not None
                else None
            ),
            git_commit=git_commit,
            notes=tuple(str(note) for note in notes),
        )
        _atomic_write_json(
            temporary / PERSONALIZATION_MANIFEST,
            manifest.to_dict(),
        )

        validation = validate_personal_model_package(
            temporary,
            verify_hashes=True,
        )
        validation.raise_for_error()

        destination.parent.mkdir(parents=True, exist_ok=True)
        _install_directory(
            temporary,
            destination,
            overwrite=overwrite,
        )
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return load_personal_model_package(
        destination,
        validate=True,
    )


def validate_personal_model_package(
    path: str | Path,
    *,
    verify_hashes: bool = True,
) -> PackageValidationResult:
    package = Path(path).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []

    if not package.is_dir():
        return PackageValidationResult(
            valid=False,
            package_path=package,
            errors=(f"Package directory does not exist: {package}",),
            warnings=(),
            runtime_ready=False,
        )

    manifest_path = package / PERSONALIZATION_MANIFEST
    if not manifest_path.is_file():
        return PackageValidationResult(
            valid=False,
            package_path=package,
            errors=(f"Missing {PERSONALIZATION_MANIFEST}.",),
            warnings=(),
            runtime_ready=False,
        )

    try:
        manifest = PersonalModelManifest.from_dict(
            _read_json(manifest_path)
        )
    except (KeyError, TypeError, ValueError) as error:
        return PackageValidationResult(
            valid=False,
            package_path=package,
            errors=(f"Invalid manifest: {error}",),
            warnings=(),
            runtime_ready=False,
        )

    classifier_path = package / manifest.classifier_path
    if not classifier_path.is_file():
        errors.append(
            f"Classifier file is missing: {manifest.classifier_path}"
        )
    elif verify_hashes:
        actual = sha256_file(classifier_path)
        if actual != manifest.classifier_sha256:
            errors.append(
                "Classifier SHA-256 mismatch: "
                f"manifest={manifest.classifier_sha256}, actual={actual}"
            )

    if manifest.personalized_backbone_path is not None:
        backbone_path = package / manifest.personalized_backbone_path
        if not backbone_path.is_file():
            errors.append(
                "Personalized backbone file is missing: "
                f"{manifest.personalized_backbone_path}"
            )
        elif verify_hashes:
            actual = sha256_file(backbone_path)
            if actual != manifest.personalized_backbone_sha256:
                errors.append(
                    "Personalized backbone SHA-256 mismatch: "
                    f"manifest={manifest.personalized_backbone_sha256}, "
                    f"actual={actual}"
                )

    base_path = Path(
        manifest.base_backbone_path
    ).expanduser()
    if not base_path.is_file():
        warnings.append(
            "Referenced base backbone is not currently available at "
            f"{base_path}. This is acceptable for a self-contained "
            "personalized-backbone package, but not for head_only runtime."
        )
    elif verify_hashes:
        actual = sha256_file(base_path)
        if actual != manifest.base_backbone_sha256:
            errors.append(
                "Base backbone SHA-256 mismatch: "
                f"manifest={manifest.base_backbone_sha256}, actual={actual}"
            )

    if manifest.runtime_ready:
        runtime_path = package / (
            manifest.runtime_package_path or "."
        )
        missing = [
            name for name in RUNTIME_REQUIRED_FILES
            if not (runtime_path / name).is_file()
        ]
        if missing:
            errors.append(
                f"Runtime package is missing required files: {missing}"
            )

    for optional_path, description in (
        (manifest.metrics_path, "metrics"),
        (manifest.training_path, "training metadata"),
    ):
        if optional_path is not None and not (
            package / optional_path
        ).is_file():
            errors.append(
                f"Declared {description} file is missing: {optional_path}"
            )

    return PackageValidationResult(
        valid=not errors,
        package_path=package,
        errors=tuple(errors),
        warnings=tuple(warnings),
        runtime_ready=manifest.runtime_ready,
    )


def load_personal_model_package(
    path: str | Path,
    *,
    validate: bool = True,
    verify_hashes: bool = True,
) -> LoadedPersonalModelPackage:
    package = Path(path).expanduser().resolve()
    manifest = PersonalModelManifest.from_dict(
        _read_json(package / PERSONALIZATION_MANIFEST)
    )
    loaded = LoadedPersonalModelPackage(
        path=package,
        manifest=manifest,
    )
    if validate:
        result = validate_personal_model_package(
            package,
            verify_hashes=verify_hashes,
        )
        result.raise_for_error()
    return loaded


def resolve_runtime_package_path(
    path: str | Path,
    *,
    validate: bool = True,
) -> Path:
    loaded = load_personal_model_package(
        path,
        validate=validate,
    )
    runtime = loaded.runtime_path
    if runtime is None:
        raise ValueError(
            f"Personal package {loaded.path} is not runtime-ready. "
            "Export a Runtime Model Package and recreate/update it."
        )
    return runtime
