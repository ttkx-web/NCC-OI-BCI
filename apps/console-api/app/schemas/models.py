from __future__ import annotations

from app.schemas.common import ConsoleModel


class ModelSummary(ConsoleModel):
    id: str
    model_name: str
    model_type: str
    head_type: str
    subject_id: str | None = None
    dataset_name: str
    task: str
    window_sec: float
    step_sec: float
    sample_rate: float
    target_channels: int
    schema_version: int
    runtime_verified: bool
    package_version: str
    balanced_accuracy: float | None = None
    macro_f1: float | None = None
    warning_message: str | None = None


class ModelList(ConsoleModel):
    items: list[ModelSummary]
