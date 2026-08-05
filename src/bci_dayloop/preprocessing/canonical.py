from __future__ import annotations

import numpy as np

from bci_dayloop.runtime.types import (
    CanonicalEEGWindow,
    RawEEGWindow,
)


CHANNEL_ALIASES = {
    "FP1": "Fp1",
    "FPZ": "Fpz",
    "FP2": "Fp2",
    "CZ": "Cz",
    "PZ": "Pz",
    "OZ": "Oz",
}


class SignalCanonicalizer:
    def __init__(self, target_unit: str = "uV") -> None:
        self.target_unit = target_unit

    def transform(self, window: RawEEGWindow) -> CanonicalEEGWindow:
        data = np.asarray(window.data)

        if data.ndim != 2:
            raise ValueError(
                f"EEG data must be 2D, but got shape={data.shape}."
            )

        history: list[str] = []

        if window.layout == "TC":
            data = data.T
            history.append("transpose_TC_to_CT")
        elif window.layout != "CT":
            raise ValueError(f"Unsupported EEG layout: {window.layout}")

        if data.shape[0] != len(window.channel_names):
            raise ValueError(
                "Channel dimension does not match channel_names: "
                f"data_channels={data.shape[0]}, "
                f"names={len(window.channel_names)}."
            )

        channel_names = [
            CHANNEL_ALIASES.get(name.strip(), name.strip())
            for name in window.channel_names
        ]
        history.append("normalize_channel_names")

        data = self._convert_unit(
            data=data,
            source_unit=window.unit,
            target_unit=self.target_unit,
        )
        history.append(
            f"convert_unit:{window.unit}->{self.target_unit}"
        )

        data = data.astype(np.float32, copy=False)
        history.append("cast_float32")

        if not np.isfinite(data).all():
            raise ValueError("EEG window contains NaN or Inf values.")

        return CanonicalEEGWindow(
            data=data,
            channel_names=channel_names,
            sample_rate=float(window.sample_rate),
            unit=self.target_unit,
            start_time_sec=window.start_time_sec,
            trial_id=window.trial_id,
            window_id=window.window_id,
            label=window.label,
            metadata=dict(window.metadata),
            processing_history=history,
        )

    @staticmethod
    def _convert_unit(
        data: np.ndarray,
        source_unit: str,
        target_unit: str,
    ) -> np.ndarray:
        normalized_source = source_unit.lower().replace("μ", "u")
        normalized_target = target_unit.lower().replace("μ", "u")

        if normalized_source == normalized_target:
            return data

        if normalized_source == "v" and normalized_target == "uv":
            return data * 1e6

        if normalized_source == "uv" and normalized_target == "v":
            return data * 1e-6

        raise ValueError(
            f"Unsupported EEG unit conversion: "
            f"{source_unit} -> {target_unit}"
        )