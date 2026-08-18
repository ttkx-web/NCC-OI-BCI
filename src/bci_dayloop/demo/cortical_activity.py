"""Fast, device-configured cortical activity visualization for the demo.

This is a sensor projection, not an inverse solution or source localization.
It overlays 1–30 Hz channel log absolute power on static lateral templates
using device-maintained 2D anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from bci_dayloop.demo.cortical_montage import CorticalMontage, canonical_channel_name, load_cortical_montage


ASSET_DIR = Path(__file__).resolve().parent / "assets" / "cortical"
CORTICAL_EMA_ALPHA = 0.85
DEFAULT_MONTAGE_NAME = "bnci_22"
DEFAULT_SIGMA_FRACTION = 0.075
ACTIVITY_THRESHOLD = 0.22


@dataclass(frozen=True, slots=True)
class CorticalActivityResult:
    left_rgba: np.ndarray
    right_rgba: np.ndarray
    update_ms: float
    available: bool
    mapped_channel_count: int
    unmapped_channel_count: int
    montage_name: str


def _load_template(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cortical asset {path}. Run tools/multistate_demo/generate_cortical_template.py once."
        )
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return rgba, rgba[..., 3].astype(np.float32) / 255.0


class CorticalActivityMapper:
    """Compose lateral overlays from named channel activity and a montage config."""

    def __init__(
        self,
        *,
        default_montage_name: str = DEFAULT_MONTAGE_NAME,
        ema_alpha: float = CORTICAL_EMA_ALPHA,
        sigma_fraction: float = DEFAULT_SIGMA_FRACTION,
    ) -> None:
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 0.99))
        self.sigma_fraction = float(np.clip(sigma_fraction, 0.01, 0.30))
        self.left_template, self.left_silhouette = _load_template(ASSET_DIR / "cortical_left_lateral.png")
        self.right_template, self.right_silhouette = _load_template(ASSET_DIR / "cortical_right_lateral.png")
        self._masks_by_montage: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        self._active_montage: CorticalMontage | None = None
        self._previous_heatmaps: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._configure_montage(default_montage_name)

    @property
    def montage_name(self) -> str:
        assert self._active_montage is not None
        return self._active_montage.name

    @property
    def _masks(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return self._masks_by_montage[self.montage_name]

    def reset(self) -> None:
        """Clear temporal EMA but retain templates and cached masks."""
        self._previous_heatmaps = {"left": None, "right": None}

    def _configure_montage(self, montage_name: str) -> None:
        montage = load_cortical_montage(montage_name)
        if self._active_montage is not None and montage.name == self._active_montage.name:
            return
        if montage.name not in self._masks_by_montage:
            self._masks_by_montage[montage.name] = self._precompute_masks(montage)
        self._active_montage = montage
        # Do not blend EMA history between different device layouts.
        self.reset()

    def _precompute_masks(self, montage: CorticalMontage) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        grids = {
            "left": np.indices(self.left_silhouette.shape, dtype=np.float32),
            "right": np.indices(self.right_silhouette.shape, dtype=np.float32),
        }
        silhouettes = {"left": self.left_silhouette, "right": self.right_silhouette}
        for name, anchors in montage.channels.items():
            by_side = {side: np.zeros_like(silhouette, dtype=np.float32) for side, silhouette in silhouettes.items()}
            for anchor in anchors:
                y_grid, x_grid = grids[anchor.hemisphere]
                height, width = silhouettes[anchor.hemisphere].shape
                sigma = max(2.0, width * self.sigma_fraction)
                center_x, center_y = anchor.x * (width - 1), anchor.y * (height - 1)
                gaussian = np.exp(-((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / (2.0 * sigma**2))
                by_side[anchor.hemisphere] += gaussian * silhouettes[anchor.hemisphere]
            masks[name] = (by_side["left"], by_side["right"])
        return masks

    @staticmethod
    def _normalize_activity(log_power: np.ndarray) -> np.ndarray:
        if log_power.size == 0:
            return log_power.astype(np.float32)
        low, high = np.percentile(log_power, (10.0, 90.0))
        if high - low < 1e-9:
            return np.full(log_power.shape, 0.55, dtype=np.float32)
        return np.clip((log_power - low) / (high - low), 0.0, 1.0).astype(np.float32)

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

    def _empty_result(self, started: float, *, unmapped: int) -> CorticalActivityResult:
        self.reset()
        return CorticalActivityResult(
            left_rgba=self.left_template.copy(),
            right_rgba=self.right_template.copy(),
            update_ms=(perf_counter() - started) * 1000.0,
            available=False,
            mapped_channel_count=0,
            unmapped_channel_count=unmapped,
            montage_name=self.montage_name,
        )

    def update(
        self,
        channel_names: list[str],
        channel_power_1_30: np.ndarray,
        *,
        channel_valid_mask: np.ndarray | None = None,
        montage_name: str | None = None,
    ) -> CorticalActivityResult:
        """Update one decoder-rate map from 1–30 Hz absolute channel power.

        Input power is in internally normalized microvolt power units. Invalid
        or unknown channels are excluded before log-percentile scaling and
        Gaussian accumulation.
        """
        started = perf_counter()
        self._configure_montage(montage_name or self.montage_name)
        power = np.asarray(channel_power_1_30, dtype=np.float64)
        if power.ndim != 1 or power.shape[0] != len(channel_names):
            raise ValueError("channel_power_1_30 must have shape [channels]")
        valid = np.ones(power.shape[0], dtype=bool) if channel_valid_mask is None else np.asarray(channel_valid_mask, dtype=bool)
        if valid.ndim != 1 or valid.shape[0] != power.shape[0]:
            raise ValueError("channel_valid_mask must have shape [channels]")
        candidates: list[tuple[str, float]] = []
        unmapped = 0
        for name, value, is_valid in zip(channel_names, power, valid, strict=True):
            canonical = canonical_channel_name(name)
            if not is_valid or not np.isfinite(value) or value < 0.0:
                continue
            if canonical not in self._masks:
                unmapped += 1
                continue
            candidates.append((canonical, float(np.log1p(value))))
        if not candidates:
            return self._empty_result(started, unmapped=unmapped)
        weights = self._normalize_activity(np.asarray([value for _, value in candidates], dtype=np.float64))
        heatmaps = {"left": np.zeros_like(self.left_silhouette), "right": np.zeros_like(self.right_silhouette)}
        for (name, _), weight in zip(candidates, weights, strict=True):
            left_mask, right_mask = self._masks[name]
            heatmaps["left"] += weight * left_mask
            heatmaps["right"] += weight * right_mask
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
            left_rgba=output["left"], right_rgba=output["right"], update_ms=(perf_counter() - started) * 1000.0,
            available=True, mapped_channel_count=len(candidates), unmapped_channel_count=unmapped, montage_name=self.montage_name,
        )


__all__ = ["ASSET_DIR", "CORTICAL_EMA_ALPHA", "DEFAULT_MONTAGE_NAME", "CorticalActivityMapper", "CorticalActivityResult"]
