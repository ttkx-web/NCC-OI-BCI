"""Formal 50M Stage-1 population training implementation."""

from __future__ import annotations

from typing import Any

__all__ = ["build_argument_parser", "main", "run_population_training"]


def __getattr__(name: str) -> Any:
    """Keep package imports light and allow ``python -m ...linear_head``."""
    if name in __all__:
        from . import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
