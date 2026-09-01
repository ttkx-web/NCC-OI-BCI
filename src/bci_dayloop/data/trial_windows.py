from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


TrialWindowAnchor = Literal["start", "center", "end"]


@dataclass(frozen=True, slots=True)
class DirectTrialSelection:
    anchor: TrialWindowAnchor
    source_samples: int
    target_samples: int
    start_sample: int
    end_sample_exclusive: int
    sample_rate: float

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "policy": "one_contiguous_window_per_source_trial",
            "anchor": self.anchor,
            "source_samples": self.source_samples,
            "source_seconds": (
                self.source_samples / self.sample_rate
            ),
            "selected_start_sample": self.start_sample,
            "selected_end_sample_exclusive": (
                self.end_sample_exclusive
            ),
            "selected_start_seconds": (
                self.start_sample / self.sample_rate
            ),
            "selected_end_seconds": (
                self.end_sample_exclusive
                / self.sample_rate
            ),
            "selected_samples": self.target_samples,
            "selected_seconds": (
                self.target_samples / self.sample_rate
            ),
        }


def select_direct_trial_window(
    data: np.ndarray,
    *,
    sample_rate: float,
    window_seconds: float,
    anchor: TrialWindowAnchor,
    context: str,
) -> tuple[np.ndarray, DirectTrialSelection]:
    """
    从每个源 trial 中选择一个连续片段。

    支持 [N, C, T] 或 [C, T] 输入。
    不 padding、不拼接、不跨 trial。
    """
    values = np.asarray(data)

    if values.ndim < 2:
        raise ValueError(
            f"{context}: expected [..., C, T] data, "
            f"got shape {values.shape}."
        )

    if sample_rate <= 0:
        raise ValueError(
            f"{context}: sample_rate must be positive, "
            f"got {sample_rate}."
        )

    if window_seconds <= 0:
        raise ValueError(
            f"{context}: window_seconds must be positive, "
            f"got {window_seconds}."
        )

    if anchor not in {"start", "center", "end"}:
        raise ValueError(
            f"{context}: unsupported anchor {anchor!r}; "
            "expected start, center, or end."
        )

    source_samples = int(values.shape[-1])
    target_samples = int(
        round(window_seconds * sample_rate)
    )

    if target_samples <= 0:
        raise ValueError(
            f"{context}: target window has no samples."
        )

    if target_samples > source_samples:
        raise ValueError(
            f"{context}: source trials are only "
            f"{source_samples / sample_rate:.3f}s, but "
            f"{window_seconds:.3f}s is requested. "
            "Direct-trial mode does not pad, concatenate, "
            "or cross source-trial boundaries."
        )

    if anchor == "start":
        start_sample = 0
    elif anchor == "center":
        start_sample = (
            source_samples - target_samples
        ) // 2
    else:
        start_sample = source_samples - target_samples

    end_sample = start_sample + target_samples

    selection = DirectTrialSelection(
        anchor=anchor,
        source_samples=source_samples,
        target_samples=target_samples,
        start_sample=start_sample,
        end_sample_exclusive=end_sample,
        sample_rate=float(sample_rate),
    )

    return (
        np.ascontiguousarray(
            values[..., start_sample:end_sample]
        ),
        selection,
    )