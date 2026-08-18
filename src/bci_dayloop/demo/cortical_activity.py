"""Fast sensor-derived cortical activity visualization for the demo layer.

This module does not perform an EEG inverse solution or source localization.
It projects per-channel sensor activity onto static lateral cortical templates
using fixed, visual-only anchors and precomputed Gaussian masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image


ASSET_DIR = Path(__file__).resolve().parent / "assets" / "cortical"
CORTICAL_EMA_ALPHA = 0.85
ACTIVITY_BANDS = ("theta", "alpha", "beta")
DEFAULT_SIGMA_FRACTION = 0.075
ACTIVITY_THRESHOLD = 0.22


@dataclass(frozen=True, slots=True)
class CorticalAnchor:
    hemisphere: str
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class CorticalActivityResult:
    left_rgba: np.ndarray
    right_rgba: np.ndarray
    update_ms: float


def _anchors(left: tuple[float, float] | None = None, right: tuple[float, float] | None = None) -> tuple[CorticalAnchor, ...]:
    anchors: list[CorticalAnchor] = []
    if left is not None:
        anchors.append(CorticalAnchor("left", *left))
    if right is not None:
        anchors.append(CorticalAnchor("right", *right))
    return tuple(anchors)


# Coordinates are normalized pixel positions in the fixed lateral templates.
# They encode a visually reasonable sensor-to-cortex projection, not anatomy or
# source estimates.  Midline electrodes intentionally contribute bilaterally.
CHANNEL_CORTICAL_POSITIONS: dict[str, tuple[CorticalAnchor, ...]] = {
    "FP1": _anchors(left=(0.18, 0.28)), "FP2": _anchors(right=(0.82, 0.28)),
    "AF3": _anchors(left=(0.23, 0.32)), "AF4": _anchors(right=(0.77, 0.32)),
    "F7": _anchors(left=(0.20, 0.42)), "F5": _anchors(left=(0.27, 0.40)), "F3": _anchors(left=(0.33, 0.39)), "F1": _anchors(left=(0.39, 0.40)),
    "F2": _anchors(right=(0.61, 0.40)), "F4": _anchors(right=(0.67, 0.39)), "F6": _anchors(right=(0.73, 0.40)), "F8": _anchors(right=(0.80, 0.42)),
    "FT7": _anchors(left=(0.29, 0.54)), "FC5": _anchors(left=(0.36, 0.47)), "FC3": _anchors(left=(0.42, 0.46)), "FC1": _anchors(left=(0.47, 0.45)),
    "FC2": _anchors(right=(0.53, 0.45)), "FC4": _anchors(right=(0.58, 0.46)), "FC6": _anchors(right=(0.64, 0.47)), "FT8": _anchors(right=(0.71, 0.54)),
    "T7": _anchors(left=(0.42, 0.68)), "C5": _anchors(left=(0.44, 0.54)), "C3": _anchors(left=(0.50, 0.52)), "C1": _anchors(left=(0.54, 0.51)),
    "C2": _anchors(right=(0.46, 0.51)), "C4": _anchors(right=(0.50, 0.52)), "C6": _anchors(right=(0.56, 0.54)), "T8": _anchors(right=(0.58, 0.68)),
    "TP7": _anchors(left=(0.51, 0.72)), "CP5": _anchors(left=(0.56, 0.59)), "CP3": _anchors(left=(0.61, 0.57)), "CP1": _anchors(left=(0.65, 0.56)),
    "CP2": _anchors(right=(0.35, 0.56)), "CP4": _anchors(right=(0.39, 0.57)), "CP6": _anchors(right=(0.44, 0.59)), "TP8": _anchors(right=(0.49, 0.72)),
    "P7": _anchors(left=(0.71, 0.70)), "P5": _anchors(left=(0.75, 0.66)), "P3": _anchors(left=(0.78, 0.63)), "P1": _anchors(left=(0.82, 0.61)),
    "P2": _anchors(right=(0.18, 0.61)), "P4": _anchors(right=(0.22, 0.63)), "P6": _anchors(right=(0.25, 0.66)), "P8": _anchors(right=(0.29, 0.70)),
    "PO3": _anchors(left=(0.87, 0.70)), "PO4": _anchors(right=(0.13, 0.70)), "O1": _anchors(left=(0.91, 0.75)), "O2": _anchors(right=(0.09, 0.75)),
    "FPZ": _anchors(left=(0.20, 0.29), right=(0.80, 0.29)), "FZ": _anchors(left=(0.36, 0.39), right=(0.64, 0.39)),
    "FCZ": _anchors(left=(0.46, 0.45), right=(0.54, 0.45)), "CZ": _anchors(left=(0.53, 0.51), right=(0.47, 0.51)),
    "CPZ": _anchors(left=(0.64, 0.56), right=(0.36, 0.56)), "PZ": _anchors(left=(0.82, 0.61), right=(0.18, 0.61)),
    "POZ": _anchors(left=(0.88, 0.70), right=(0.12, 0.70)), "OZ": _anchors(left=(0.92, 0.75), right=(0.08, 0.75)),
}


def _canonical_channel_name(name: str) -> str:
    normalized = name.upper().strip().replace(" ", "")
    return normalized.removeprefix("EEG-").removeprefix("EEG")


def _load_template(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cortical asset {path}. Run tools/multistate_demo/generate_cortical_template.py once."
        )
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return rgba, rgba[..., 3].astype(np.float32) / 255.0


class CorticalActivityMapper:
    """Compose two lateral cortical overlays with precomputed Gaussian masks."""

    def __init__(self, *, ema_alpha: float = CORTICAL_EMA_ALPHA, sigma_fraction: float = DEFAULT_SIGMA_FRACTION) -> None:
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 0.99))
        self.left_template, self.left_silhouette = _load_template(ASSET_DIR / "cortical_left_lateral.png")
        self.right_template, self.right_silhouette = _load_template(ASSET_DIR / "cortical_right_lateral.png")
        self._masks = self._precompute_masks(float(sigma_fraction))
        self._previous_heatmaps: dict[str, np.ndarray | None] = {"left": None, "right": None}

    def reset(self) -> None:
        """Reset only temporal EMA; static templates and masks stay cached."""
        self._previous_heatmaps = {"left": None, "right": None}

    def _precompute_masks(self, sigma_fraction: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        grids = {
            "left": np.indices(self.left_silhouette.shape, dtype=np.float32),
            "right": np.indices(self.right_silhouette.shape, dtype=np.float32),
        }
        silhouettes = {"left": self.left_silhouette, "right": self.right_silhouette}
        for name, anchors in CHANNEL_CORTICAL_POSITIONS.items():
            by_side = {side: np.zeros_like(silhouette, dtype=np.float32) for side, silhouette in silhouettes.items()}
            for anchor in anchors:
                y_grid, x_grid = grids[anchor.hemisphere]
                height, width = silhouettes[anchor.hemisphere].shape
                sigma = max(2.0, width * sigma_fraction)
                center_x, center_y = anchor.x * (width - 1), anchor.y * (height - 1)
                gaussian = np.exp(-((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / (2.0 * sigma**2))
                by_side[anchor.hemisphere] += gaussian * silhouettes[anchor.hemisphere]
            masks[name] = (by_side["left"], by_side["right"])
        return masks

    @staticmethod
    def _normalize_activity(activity: np.ndarray) -> np.ndarray:
        if activity.size == 0:
            return activity
        low, high = np.percentile(activity, (10.0, 90.0))
        if high - low < 1e-9:
            return np.full(activity.shape, 0.55, dtype=np.float32)
        return np.clip((activity - low) / (high - low), 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _normalize_heatmap(heatmap: np.ndarray, silhouette: np.ndarray) -> np.ndarray:
        visible = heatmap[(heatmap > 1e-7) & (silhouette > 0.0)]
        if visible.size == 0:
            return np.zeros_like(heatmap, dtype=np.float32)
        scale = float(np.percentile(visible, 92.0))
        return np.clip(heatmap / max(scale, 1e-8), 0.0, 1.0).astype(np.float32) * silhouette

    @staticmethod
    def _compose(template: np.ndarray, silhouette: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        intensity = np.clip((heatmap - ACTIVITY_THRESHOLD) / (1.0 - ACTIVITY_THRESHOLD), 0.0, 1.0)
        alpha = (intensity**0.75) * 0.72 * silhouette
        overlay = np.empty((*heatmap.shape, 3), dtype=np.float32)
        overlay[..., 0] = 1.0
        overlay[..., 1] = 0.10 + 0.88 * intensity
        overlay[..., 2] = 0.015
        base = template[..., :3].astype(np.float32) / 255.0
        composed = base * (1.0 - alpha[..., None]) + overlay * alpha[..., None]
        return np.dstack((np.rint(composed * 255.0).astype(np.uint8), template[..., 3]))

    def update(self, channel_names: list[str], channel_relative_band_power: dict[str, np.ndarray]) -> CorticalActivityResult:
        """Map existing theta/alpha/beta channel power onto the static templates."""
        started = perf_counter()
        if not all(band in channel_relative_band_power for band in ACTIVITY_BANDS):
            activity = np.empty(0, dtype=np.float32)
            supported: list[str] = []
        else:
            activity = np.sum(np.vstack([channel_relative_band_power[band] for band in ACTIVITY_BANDS]), axis=0)
            supported = [
                _canonical_channel_name(name)
                for name in channel_names
            ]
        normalized = self._normalize_activity(np.asarray(activity, dtype=np.float32))
        heatmaps = {"left": np.zeros_like(self.left_silhouette), "right": np.zeros_like(self.right_silhouette)}
        for index, name in enumerate(supported):
            if index >= normalized.size or name not in self._masks:
                continue
            left_mask, right_mask = self._masks[name]
            heatmaps["left"] += normalized[index] * left_mask
            heatmaps["right"] += normalized[index] * right_mask
        templates = {"left": self.left_template, "right": self.right_template}
        silhouettes = {"left": self.left_silhouette, "right": self.right_silhouette}
        output: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            current = self._normalize_heatmap(heatmaps[side], silhouettes[side])
            previous = self._previous_heatmaps[side]
            smoothed = current if previous is None else self.ema_alpha * previous + (1.0 - self.ema_alpha) * current
            self._previous_heatmaps[side] = smoothed
            output[side] = self._compose(templates[side], silhouettes[side], smoothed)
        return CorticalActivityResult(
            left_rgba=output["left"], right_rgba=output["right"], update_ms=(perf_counter() - started) * 1000.0
        )


__all__ = [
    "ACTIVITY_BANDS",
    "ASSET_DIR",
    "CHANNEL_CORTICAL_POSITIONS",
    "CORTICAL_EMA_ALPHA",
    "CorticalActivityMapper",
    "CorticalActivityResult",
]
