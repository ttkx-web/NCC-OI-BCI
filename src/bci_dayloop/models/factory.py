from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from bci_dayloop.models.base import BaseModelAdapter

ModelBuilder = Callable[..., BaseModelAdapter]


class ModelFactory:
    _registry: dict[str, ModelBuilder] = {}

    @classmethod
    def register(cls, name: str, builder: ModelBuilder) -> None:
        cls._registry[name] = builder

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseModelAdapter:
        if name not in cls._registry:
            raise ValueError(f"Unknown model '{name}'. Available: {', '.join(cls.list_models())}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list_models(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def load_package(cls, path: str | Path, device: str = "cpu") -> BaseModelAdapter:
        from bci_dayloop.utils.config import load_yaml

        config = load_yaml(Path(path) / "model.yaml")
        name = str(config["name"])
        if name != "labram-linear":
            raise ValueError(f"Unsupported model package '{name}'")
        from bci_dayloop.models.labram_linear import LaBraMLinearAdapter

        return LaBraMLinearAdapter.from_package(path, device=device)


def register_default_models() -> None:
    if "labram-linear" not in ModelFactory._registry:
        from bci_dayloop.models.labram_linear import LaBraMLinearAdapter

        ModelFactory.register("labram-linear", LaBraMLinearAdapter)


register_default_models()

