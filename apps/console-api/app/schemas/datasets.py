from __future__ import annotations

from app.schemas.common import ConsoleModel


class DatasetSummary(ConsoleModel):
    id: str
    name: str
    subject_id: str
    sessions: list[str]
    trial_count: int
    channel_count: int
    sample_rate: float
    unit: str
    class_names: list[str]
    qc_status: str


class DatasetList(ConsoleModel):
    items: list[DatasetSummary]
