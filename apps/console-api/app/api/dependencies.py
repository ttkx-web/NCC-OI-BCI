from __future__ import annotations

from fastapi import Request

from app.services.dataset_service import DatasetRegistry
from app.services.model_service import ModelRegistry
from app.services.run_service import RunService


def model_registry(request: Request) -> ModelRegistry:
    return request.app.state.models


def dataset_registry(request: Request) -> DatasetRegistry:
    return request.app.state.datasets


def run_service(request: Request) -> RunService:
    return request.app.state.runs

