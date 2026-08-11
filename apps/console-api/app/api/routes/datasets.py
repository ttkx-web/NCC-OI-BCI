from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import dataset_registry
from app.schemas.datasets import DatasetList, DatasetSummary
from app.services.dataset_service import DatasetRegistry


router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=DatasetList)
def list_datasets(registry: Annotated[DatasetRegistry, Depends(dataset_registry)]) -> DatasetList:
    return DatasetList(items=registry.list())


@router.get("/{dataset_id}", response_model=DatasetSummary)
def get_dataset(
    dataset_id: str,
    registry: Annotated[DatasetRegistry, Depends(dataset_registry)],
    subject: str | None = Query(default=None),
) -> DatasetSummary:
    try:
        return registry.get_entry(dataset_id, subject).summary
    except LookupError as error:
        raise HTTPException(status_code=404, detail="Dataset not found or subject is ambiguous") from error
