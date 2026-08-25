"""Training entry points, imported lazily to keep optional ML dependencies optional."""

__all__ = ["train_linear_probe"]


def __getattr__(name: str):
    if name == "train_linear_probe":
        from .pipeline import train_linear_probe
        return train_linear_probe
    raise AttributeError(name)
