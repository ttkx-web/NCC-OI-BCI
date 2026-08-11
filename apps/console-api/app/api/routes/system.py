from __future__ import annotations

from fastapi import APIRouter

from app.services.runtime_service import system_status


router = APIRouter(prefix="/system", tags=["system"])


@router.get("/status")
def status() -> dict[str, object]:
    return system_status()

