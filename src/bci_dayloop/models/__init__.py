from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BaseModelAdapter": (".base", "BaseModelAdapter"),
    "ModelFactory": (".factory", "ModelFactory"),
    "register_default_models": (".factory", "register_default_models"),
    "LaBraMLinearAdapter": (".labram_linear", "LaBraMLinearAdapter"),
}


__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
