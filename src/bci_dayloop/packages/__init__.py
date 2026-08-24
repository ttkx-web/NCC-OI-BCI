from .loader import (
    LoadedRuntimePackage,
    load_runtime_package,
)
__all__ = [
    "LoadedRuntimePackage",
    "load_runtime_package",
    "export_50m_multi_head_runtime_package",
    "load_multi_head_runtime_package",
]


def __getattr__(name: str):
    """Avoid importing the application package while common helpers initialize."""
    if name in {"export_50m_multi_head_runtime_package", "load_multi_head_runtime_package"}:
        from .multi_head import (
            export_50m_multi_head_runtime_package,
            load_multi_head_runtime_package,
        )
        return {
            "export_50m_multi_head_runtime_package": export_50m_multi_head_runtime_package,
            "load_multi_head_runtime_package": load_multi_head_runtime_package,
        }[name]
    raise AttributeError(name)
