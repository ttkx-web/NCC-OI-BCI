from __future__ import annotations

from enum import Enum

from pydantic import Field

from app.schemas.common import ConsoleModel


class ComputeDevice(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class RunState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


class ReplayCreate(ConsoleModel):
    dataset_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    session: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    compute_device: ComputeDevice = ComputeDevice.CPU
    replay_speed: float = Field(default=1.0, gt=0, le=100)
    max_windows: int = Field(default=100, gt=0, le=10000)
    confidence_threshold: float = Field(default=0.55, ge=0, le=1)


class RunCreated(ConsoleModel):
    run_id: str
    state: RunState


class RunSummary(ConsoleModel):
    id: str
    run_type: str
    state: RunState
    dataset_id: str
    subject_id: str
    session: str
    model_id: str
    created_at: float
    successful_windows: int = 0
    failed_windows: int = 0
    expected_windows: int | None = None


class RunList(ConsoleModel):
    items: list[RunSummary]
