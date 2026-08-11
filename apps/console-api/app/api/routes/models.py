from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import model_registry
from app.schemas.models import ModelList, ModelSummary
from app.services.model_service import ModelRegistry


router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelList)
def list_models(
    registry: Annotated[ModelRegistry, Depends(model_registry)],
    backbone: str | None = Query(default=None),
    subject: str | None = Query(default=None),
    adaptation: str | None = Query(default=None),
) -> ModelList:
    return ModelList(items=registry.list(backbone=backbone, subject=subject, adaptation=adaptation))


@router.get("/{model_id}", response_model=ModelSummary)
def get_model(model_id: str, registry: Annotated[ModelRegistry, Depends(model_registry)]) -> ModelSummary:
    try:
        return registry.get_entry(model_id).summary
    except LookupError as error:
        raise HTTPException(status_code=404, detail="Model not found") from error

