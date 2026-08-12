from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import run_service
from app.schemas.runs import LiveCreate, ReplayCreate, RunCreated, RunList, RunState, RunSummary
from app.services.run_service import RunService


router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("/replay", response_model=RunCreated, status_code=202)
def create_replay(payload: ReplayCreate, service: Annotated[RunService, Depends(run_service)]) -> RunCreated:
    try:
        record = service.create_replay(payload)
    except LookupError as error:
        raise HTTPException(status_code=404, detail="Dataset or model not found") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return RunCreated(run_id=record.id, state=RunState.STARTING)


@router.post("/live", response_model=RunCreated, status_code=202)
def create_live(payload: LiveCreate, service: Annotated[RunService, Depends(run_service)]) -> RunCreated:
    try:
        record = service.create_live(payload)
    except LookupError as error:
        raise HTTPException(status_code=404, detail="Model not found") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return RunCreated(run_id=record.id, state=RunState.STARTING)


@router.get("", response_model=RunList)
def list_runs(service: Annotated[RunService, Depends(run_service)]) -> RunList:
    return RunList(items=service.list())


@router.get("/{run_id}", response_model=RunSummary)
def get_run(run_id: str, service: Annotated[RunService, Depends(run_service)]) -> RunSummary:
    try:
        return service.get(run_id).summary()
    except LookupError as error:
        raise HTTPException(status_code=404, detail="Run not found") from error


@router.post("/{run_id}/stop", response_model=RunSummary)
def stop_run(run_id: str, service: Annotated[RunService, Depends(run_service)]) -> RunSummary:
    try:
        return service.stop(run_id).summary()
    except LookupError as error:
        raise HTTPException(status_code=404, detail="Run not found") from error
    except (RuntimeError, TimeoutError) as error:
        raise HTTPException(status_code=409, detail="Run could not be stopped") from error


@router.post("/{run_id}/restart", response_model=RunSummary)
def restart_run(run_id: str, service: Annotated[RunService, Depends(run_service)]) -> RunSummary:
    try:
        return service.restart(run_id).summary()
    except LookupError as error:
        raise HTTPException(status_code=404, detail="Run not found") from error
    except (RuntimeError, TimeoutError) as error:
        raise HTTPException(status_code=409, detail="Run is not ready to restart") from error
