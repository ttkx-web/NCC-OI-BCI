from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_root(
    environment_name: str,
    default_path: Path,
) -> Path:
    """Resolve a storage root, allowing environment overrides."""
    value = os.environ.get(environment_name)

    if value:
        return Path(value).expanduser().resolve()

    return default_path.resolve()


DATA_ROOT = _resolve_root(
    "BCI_DATA_ROOT",
    PROJECT_ROOT / "data",
)

CHECKPOINT_ROOT = _resolve_root(
    "BCI_CHECKPOINT_ROOT",
    PROJECT_ROOT / "checkpoints",
)

RUN_ROOT = _resolve_root(
    "BCI_RUN_ROOT",
    PROJECT_ROOT / "runs",
)

MODEL_PACKAGE_ROOT = _resolve_root(
    "BCI_MODEL_PACKAGE_ROOT",
    PROJECT_ROOT / "model_packages",
)

REGISTRY_ROOT = _resolve_root(
    "BCI_REGISTRY_ROOT",
    PROJECT_ROOT / "registries",
)


def subject_tag(subject_id: int) -> str:
    if subject_id <= 0:
        raise ValueError("subject_id must be positive")

    return f"subject_{subject_id:02d}"


def window_tag(window_seconds: float) -> str:
    value = float(window_seconds)

    if value <= 0:
        raise ValueError("window_seconds must be positive")

    if value.is_integer():
        return f"{int(value)}s"

    normalized = str(value).replace(".", "p")
    return f"{normalized}s"


def contract_tag(
    window_seconds: float,
    aggregation: str,
) -> str:
    normalized_aggregation = (
        aggregation.strip().lower().replace(" ", "_")
    )

    return (
        f"{window_tag(window_seconds)}_"
        f"{normalized_aggregation}"
    )


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def processed_subject_path(
    dataset: str,
    subject_id: int,
) -> Path:
    return (
        DATA_ROOT
        / "processed"
        / dataset
        / f"{subject_tag(subject_id)}.h5"
    )


def backbone_checkpoint_path(
    model_name: str,
    filename: str,
) -> Path:
    return (
        CHECKPOINT_ROOT
        / "backbones"
        / model_name
        / filename
    )


def population_head_path(
    *,
    stage: str,
    dataset: str,
    subject_id: int,
    window_seconds: float,
    aggregation: str,
) -> Path:
    return (
        CHECKPOINT_ROOT
        / "heads"
        / stage
        / dataset
        / subject_tag(subject_id)
        / "population"
        / contract_tag(
            window_seconds,
            aggregation,
        )
        / "head.pt"
    )


def personal_head_path(
    *,
    stage: str,
    dataset: str,
    subject_id: int,
    window_seconds: float,
    aggregation: str,
    trials_per_class: int,
    seed: int,
) -> Path:
    return (
        CHECKPOINT_ROOT
        / "heads"
        / stage
        / dataset
        / subject_tag(subject_id)
        / "personal"
        / contract_tag(
            window_seconds,
            aggregation,
        )
        / f"trials_{trials_per_class}"
        / f"seed_{seed}"
        / "head.pt"
    )


def population_run_dir(
    *,
    stage: str,
    dataset: str,
    subject_id: int,
    window_seconds: float,
    aggregation: str,
    run_id: str | None = None,
) -> Path:
    return (
        RUN_ROOT
        / stage
        / dataset
        / subject_tag(subject_id)
        / "population"
        / contract_tag(
            window_seconds,
            aggregation,
        )
        / (run_id or timestamp_id())
    )


def personal_run_dir(
    *,
    stage: str,
    dataset: str,
    subject_id: int,
    window_seconds: float,
    aggregation: str,
    trials_per_class: int,
    seed: int,
    run_id: str | None = None,
) -> Path:
    return (
        RUN_ROOT
        / stage
        / dataset
        / subject_tag(subject_id)
        / "personal"
        / contract_tag(
            window_seconds,
            aggregation,
        )
        / f"trials_{trials_per_class}"
        / f"seed_{seed}"
        / (run_id or timestamp_id())
    )


def comparison_run_dir(
    *,
    stage: str,
    dataset: str,
    subject_id: int,
    session: str,
    run_id: str | None = None,
) -> Path:
    return (
        RUN_ROOT
        / stage
        / dataset
        / subject_tag(subject_id)
        / "comparisons"
        / session
        / (run_id or timestamp_id())
    )


def population_package_dir(
    *,
    stage: str,
    dataset: str,
    subject_id: int,
    window_seconds: float,
    aggregation: str,
    version: str,
) -> Path:
    return (
        MODEL_PACKAGE_ROOT
        / stage
        / dataset
        / subject_tag(subject_id)
        / "population"
        / contract_tag(
            window_seconds,
            aggregation,
        )
        / version
    )


def personal_package_dir(
    *,
    stage: str,
    dataset: str,
    subject_id: int,
    window_seconds: float,
    aggregation: str,
    trials_per_class: int,
    seed: int,
    version: str,
) -> Path:
    return (
        MODEL_PACKAGE_ROOT
        / stage
        / dataset
        / subject_tag(subject_id)
        / "personal"
        / contract_tag(
            window_seconds,
            aggregation,
        )
        / f"trials_{trials_per_class}"
        / f"seed_{seed}"
        / version
    )


def personal_registry_path(
    stage: str = "stage1",
) -> Path:
    return (
        REGISTRY_ROOT
        / f"{stage}_personal_models.json"
    )