"""Small, dependency-light helpers used only by the multi-state demo."""

from __future__ import annotations

from collections import deque
import numpy as np


def bounded_score(value: float, *, center: float, scale: float) -> float:
    """Map an unbounded EEG feature to a stable 0--100 display score."""
    return float(np.clip(50.0 + 50.0 * np.tanh((value - center) / max(scale, 1e-9)), 0.0, 100.0))


def score_level(value: float) -> str:
    if value < 30.0:
        return "低"
    if value < 55.0:
        return "中等"
    if value < 75.0:
        return "较高"
    return "高"


class RollingLatency:
    def __init__(self, limit: int = 120) -> None:
        self.values: deque[float] = deque(maxlen=limit)

    def add(self, value: float) -> tuple[float, float]:
        self.values.append(float(value))
        array = np.asarray(self.values, dtype=np.float64)
        return float(array.mean()), float(np.percentile(array, 95))


def standard_1020_positions(channel_names: list[str]) -> tuple[np.ndarray | None, list[str] | None]:
    """Resolve common 10-20/10-10 xy coordinates without MNE global state.

    The UI still prefers ``mne.viz.plot_topomap`` when MNE is usable.  Keeping
    this lookup local makes feature extraction work in headless deployments
    where MNE cannot write its optional user configuration file.
    """
    coordinates = {
        "fp1": (-0.32, 0.94), "fpz": (0.0, 0.98), "fp2": (0.32, 0.94),
        "af3": (-0.30, 0.78), "af4": (0.30, 0.78),
        "f7": (-0.72, 0.52), "f5": (-0.48, 0.53), "f3": (-0.28, 0.55), "f1": (-0.10, 0.56), "fz": (0.0, 0.58), "f2": (0.10, 0.56), "f4": (0.28, 0.55), "f6": (0.48, 0.53), "f8": (0.72, 0.52),
        "ft7": (-0.84, 0.25), "fc5": (-0.58, 0.28), "fc3": (-0.34, 0.29), "fc1": (-0.12, 0.30), "fcz": (0.0, 0.31), "fc2": (0.12, 0.30), "fc4": (0.34, 0.29), "fc6": (0.58, 0.28), "ft8": (0.84, 0.25),
        "t7": (-0.92, 0.0), "c5": (-0.62, 0.0), "c3": (-0.36, 0.0), "c1": (-0.12, 0.0), "cz": (0.0, 0.0), "c2": (0.12, 0.0), "c4": (0.36, 0.0), "c6": (0.62, 0.0), "t8": (0.92, 0.0),
        "tp7": (-0.84, -0.25), "cp5": (-0.58, -0.28), "cp3": (-0.34, -0.29), "cp1": (-0.12, -0.30), "cpz": (0.0, -0.31), "cp2": (0.12, -0.30), "cp4": (0.34, -0.29), "cp6": (0.58, -0.28), "tp8": (0.84, -0.25),
        "p7": (-0.72, -0.52), "p5": (-0.48, -0.53), "p3": (-0.28, -0.55), "p1": (-0.10, -0.56), "pz": (0.0, -0.58), "p2": (0.10, -0.56), "p4": (0.28, -0.55), "p6": (0.48, -0.53), "p8": (0.72, -0.52),
        "po3": (-0.30, -0.78), "poz": (0.0, -0.82), "po4": (0.30, -0.78),
        "o1": (-0.32, -0.94), "oz": (0.0, -0.98), "o2": (0.32, -0.94),
    }
    valid = [(name, coordinates[name.casefold()]) for name in channel_names if name.casefold() in coordinates]
    if len(valid) < 3:
        return None, None
    return np.asarray([position for _, position in valid], dtype=np.float64), [name for name, _ in valid]
