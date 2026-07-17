from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bci_dayloop.acquisition.base import AbstractAcquirer

AcquirerBuilder = Callable[..., AbstractAcquirer]


class AcquirerFactory:
    _registry: dict[str, AcquirerBuilder] = {}

    @classmethod
    def register(cls, name: str, builder: AcquirerBuilder) -> None:
        cls._registry[name] = builder

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> AbstractAcquirer:
        if name not in cls._registry:
            raise ValueError(f"Unknown acquirer '{name}'. Available: {', '.join(cls.list_acquirers())}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list_acquirers(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def list_devices(cls) -> list[str]:
        """Compatibility alias for the reference project's Factory API."""
        return cls.list_acquirers()


def register_default_acquirers() -> None:
    if "replay" not in AcquirerFactory._registry:
        from bci_dayloop.acquisition.replay import ReplayAcquirer

        AcquirerFactory.register("replay", ReplayAcquirer)


register_default_acquirers()

