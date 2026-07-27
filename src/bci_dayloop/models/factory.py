from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from bci_dayloop.models.base import BaseModelAdapter

ModelBuilder = Callable[..., BaseModelAdapter]
PackageLoader = Callable[..., BaseModelAdapter]


class ModelFactory:
    _registry: dict[str, ModelBuilder] = {}
    _package_loaders: dict[str, PackageLoader] = {}

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
    def register_package_loader(cls, name: str, loader: PackageLoader) -> None:
        cls._package_loaders[name] = loader

    @classmethod
    def list_package_loaders(cls) -> list[str]:
        return sorted(cls._package_loaders)

    @classmethod
    def load_package(cls, path: str | Path, device: str = "cpu") -> BaseModelAdapter:
        from bci_dayloop.utils.config import load_yaml

        config = load_yaml(Path(path) / "model.yaml")
        name = str(config["name"])
        if name not in cls._package_loaders:
            raise ValueError(
                f"Unknown model package '{name}'. Available: {', '.join(cls.list_package_loaders())}"
            )
        return cls._package_loaders[name](path, device=device)


def register_default_models() -> None:
    if "labram-linear" not in ModelFactory._registry or "labram-linear" not in ModelFactory._package_loaders:
        from bci_dayloop.models.labram_linear import LaBraMLinearAdapter

        if "labram-linear" not in ModelFactory._registry:
            ModelFactory.register("labram-linear", LaBraMLinearAdapter)
        if "labram-linear" not in ModelFactory._package_loaders:
            ModelFactory.register_package_loader("labram-linear", LaBraMLinearAdapter.from_package)


register_default_models()

