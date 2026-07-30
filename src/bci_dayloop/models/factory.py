from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from bci_dayloop.models.base import BaseModelAdapter

ModelBuilder = Callable[..., BaseModelAdapter]
PackageLoader = Callable[..., BaseModelAdapter]
RuntimeLoader = Callable[..., Any]


class ModelFactory:
    _registry: dict[str, ModelBuilder] = {}
    _package_loaders: dict[str, PackageLoader] = {}
    _runtime_loaders: dict[str, RuntimeLoader] = {}

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

    @classmethod
    def register_runtime_loader(cls, name: str, loader: RuntimeLoader) -> None:
        cls._runtime_loaders[name] = loader

    @classmethod
    def list_runtime_loaders(cls) -> list[str]:
        return sorted(cls._runtime_loaders)

    @classmethod
    def load_runtime_package(cls, package_path: str | Path, metadata: Any, device: str = "cpu") -> Any:
        from bci_dayloop.utils.config import load_yaml

        package = Path(package_path)
        name = str(load_yaml(package / "model.yaml")["name"])
        if name not in cls._runtime_loaders:
            raise ValueError(f"Unknown model runtime package '{name}'. Available: {', '.join(cls.list_runtime_loaders())}")
        return cls._runtime_loaders[name](package, metadata, device=device)


def register_default_models() -> None:
    if "labram-linear" not in ModelFactory._registry or "labram-linear" not in ModelFactory._package_loaders:
        from bci_dayloop.models.labram_linear import LaBraMLinearAdapter

        if "labram-linear" not in ModelFactory._registry:
            ModelFactory.register("labram-linear", LaBraMLinearAdapter)
        if "labram-linear" not in ModelFactory._package_loaders:
            ModelFactory.register_package_loader("labram-linear", LaBraMLinearAdapter.from_package)
    if "50m-linear" not in ModelFactory._registry or "50m-linear" not in ModelFactory._package_loaders:
        try:
            from bci_dayloop.models.model_50m.adapter import Model50MAdapter
        except ImportError as import_error:
            def unavailable_50m_adapter(
                *args: Any,
                _import_error: ImportError = import_error,
                **kwargs: Any,
            ) -> BaseModelAdapter:
                raise RuntimeError(
                    "50m-linear is unavailable because its optional dependencies could not be imported"
                ) from _import_error

            if "50m-linear" not in ModelFactory._registry:
                ModelFactory.register("50m-linear", unavailable_50m_adapter)
            if "50m-linear" not in ModelFactory._package_loaders:
                ModelFactory.register_package_loader("50m-linear", unavailable_50m_adapter)
        else:
            if "50m-linear" not in ModelFactory._registry:
                ModelFactory.register("50m-linear", Model50MAdapter)
            if "50m-linear" not in ModelFactory._package_loaders:
                ModelFactory.register_package_loader("50m-linear", Model50MAdapter.from_package)
    from bci_dayloop.models.runtime_package import load_50m_runtime_package, load_labram_runtime_package
    if "labram-linear" not in ModelFactory._runtime_loaders:
        ModelFactory.register_runtime_loader("labram-linear", load_labram_runtime_package)
    if "50m-linear" not in ModelFactory._runtime_loaders:
        ModelFactory.register_runtime_loader("50m-linear", load_50m_runtime_package)


register_default_models()

