"""Display-only interpolation state for the high-frequency demo fragment.

The state intentionally keeps EEG feature/decoder results immutable. It only
interpolates PSD display targets between decode ticks; cortical images update
atomically with the decoder result, whose mapper already applies decode-time
EMA smoothing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import numpy as np


VISUAL_TARGET_FPS = 15.0
VISUAL_INTERVAL_SEC = 1.0 / VISUAL_TARGET_FPS
MAX_VISUAL_DT_SEC = 0.20


@dataclass(slots=True)
class VisualState:
    """Wall-clock-aware visual cursor and interpolated display arrays."""

    stream_time_sec: float = 0.0
    last_visual_wall_time: float | None = None
    last_visual_dt_sec: float = 0.0
    previous_psd: np.ndarray | None = None
    target_psd: np.ndarray | None = None
    displayed_psd: np.ndarray | None = None
    psd_transition_stream_time: float = 0.0
    target_cortical_left: np.ndarray | None = None
    target_cortical_right: np.ndarray | None = None
    displayed_cortical_left: np.ndarray | None = None
    displayed_cortical_right: np.ndarray | None = None
    visual_intervals: deque[float] = field(default_factory=lambda: deque(maxlen=60))
    visual_tick_count: int = 0
    decode_tick_count: int = 0
    waveform_render_ms: float = 0.0
    psd_render_ms: float = 0.0
    cortical_render_ms: float = 0.0

    def reset(self) -> None:
        self.stream_time_sec = 0.0
        self.last_visual_wall_time = None
        self.last_visual_dt_sec = 0.0
        self.previous_psd = None
        self.target_psd = None
        self.displayed_psd = None
        self.psd_transition_stream_time = 0.0
        self.target_cortical_left = None
        self.target_cortical_right = None
        self.displayed_cortical_left = None
        self.displayed_cortical_right = None
        self.visual_intervals.clear()
        self.visual_tick_count = 0
        self.decode_tick_count = 0
        self.waveform_render_ms = 0.0
        self.psd_render_ms = 0.0
        self.cortical_render_ms = 0.0

    def pause(self) -> None:
        self.last_visual_wall_time = None
        self.last_visual_dt_sec = 0.0

    def advance(self, now: float, playback_speed: float) -> float:
        """Advance stream time using measured wall time, with a safe catch-up cap."""
        if self.last_visual_wall_time is None:
            self.last_visual_wall_time = now
            self.last_visual_dt_sec = 0.0
            self.visual_tick_count += 1
            return 0.0
        raw_dt = max(0.0, now - self.last_visual_wall_time)
        self.last_visual_wall_time = now
        if raw_dt > 0.0:
            self.visual_intervals.append(raw_dt)
        self.last_visual_dt_sec = min(raw_dt, MAX_VISUAL_DT_SEC)
        self.stream_time_sec += self.last_visual_dt_sec * playback_speed
        self.visual_tick_count += 1
        return self.last_visual_dt_sec

    @property
    def median_visual_fps(self) -> float | None:
        if not self.visual_intervals:
            return None
        median_dt = float(np.median(np.asarray(self.visual_intervals, dtype=np.float64)))
        return 1.0 / median_dt if median_dt > 0.0 else None

    def set_decode_targets(self, psd: np.ndarray | None, cortical_left: np.ndarray | None, cortical_right: np.ndarray | None) -> None:
        """Install a new immutable decode target without modifying its result."""
        self.decode_tick_count += 1
        if psd is not None:
            target = np.asarray(psd, dtype=np.float32)
            if self.displayed_psd is None or self.displayed_psd.shape != target.shape:
                self.displayed_psd = target.copy()
                self.previous_psd = target.copy()
            else:
                self.previous_psd = self.displayed_psd.copy()
            self.target_psd = target.copy()
            self.psd_transition_stream_time = self.stream_time_sec
        self._set_cortical_target("left", cortical_left)
        self._set_cortical_target("right", cortical_right)

    def _set_cortical_target(self, side: str, image: np.ndarray | None) -> None:
        if image is None:
            return
        target = np.asarray(image, dtype=np.float32)
        display_name = f"displayed_cortical_{side}"
        target_name = f"target_cortical_{side}"
        # Cortical mapper output is already EMA-smoothed at decode rate.  Do
        # not add a second visual-rate filter: it both delays target response
        # and forces a high-frequency image replacement that can flicker.
        setattr(self, display_name, target.copy())
        setattr(self, target_name, target.copy())

    def interpolate(self, *, decode_interval_sec: float) -> None:
        """Interpolate PSD linearly between decoder updates."""
        if self.target_psd is not None and self.previous_psd is not None:
            progress = np.clip(
                (self.stream_time_sec - self.psd_transition_stream_time) / max(decode_interval_sec, 1e-6),
                0.0,
                1.0,
            )
            self.displayed_psd = (1.0 - progress) * self.previous_psd + progress * self.target_psd
    def displayed_cortical_rgba(self, side: str) -> np.ndarray | None:
        image = getattr(self, f"displayed_cortical_{side}")
        return None if image is None else np.clip(image, 0.0, 255.0).astype(np.uint8)


__all__ = [
    "MAX_VISUAL_DT_SEC",
    "VISUAL_INTERVAL_SEC",
    "VISUAL_TARGET_FPS",
    "VisualState",
]
