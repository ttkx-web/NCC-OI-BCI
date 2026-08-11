from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import dataset_registry
from app.services.dataset_service import DatasetRegistry


router = APIRouter(prefix="/subjects", tags=["subjects"])


@router.get("")
def list_subjects(registry: Annotated[DatasetRegistry, Depends(dataset_registry)]) -> dict[str, object]:
    subjects = sorted({item.subject_id for item in registry.list()})
    return {"items": [{"id": subject, "display_name": f"被试 {subject}"} for subject in subjects]}

