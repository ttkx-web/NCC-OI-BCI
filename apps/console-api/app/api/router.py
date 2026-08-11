from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import datasets, experiments, health, models, runs, subjects, system


api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(system.router)
api_router.include_router(models.router)
api_router.include_router(datasets.router)
api_router.include_router(subjects.router)
api_router.include_router(runs.router)
api_router.include_router(experiments.router)

