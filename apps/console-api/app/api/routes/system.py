from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import run_service
from app.services.runtime_service import system_status
from app.services.run_service import RunService


router = APIRouter(prefix="/system", tags=["system"])


@router.get("/status")
def status(service: Annotated[RunService, Depends(run_service)]) -> dict[str, object]:
    return system_status(service)
