from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.config import settings
from app.services.dataset_service import DatasetRegistry
from app.services.model_service import ModelRegistry
from app.services.run_service import RunService
from app.websocket.run_stream import stream_run


@asynccontextmanager
async def lifespan(application: FastAPI):
    models = ModelRegistry(settings.model_roots)
    datasets = DatasetRegistry(settings.dataset_root)
    application.state.models = models
    application.state.datasets = datasets
    application.state.runs = RunService(datasets, models)
    yield


app = FastAPI(
    title="NCC BCI Console API",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
app.include_router(api_router, prefix=settings.api_prefix)


@app.websocket(f"{settings.websocket_prefix}/runs/{{run_id}}")
async def run_websocket(websocket: WebSocket, run_id: str) -> None:
    await stream_run(websocket, run_id, websocket.app.state.runs)
