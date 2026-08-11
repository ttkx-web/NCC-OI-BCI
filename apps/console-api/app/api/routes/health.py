from __future__ import annotations

from fastapi import APIRouter

from app.services.runtime_service import runtime_available


router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "api_version": "v1", "runtime_available": runtime_available()}
