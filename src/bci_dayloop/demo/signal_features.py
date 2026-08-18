"""Traditional, lightweight EEG feature extraction for the demo layer."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np


EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass(frozen=True, slots=True)
class SignalFeatures:
    frequencies: np.ndarray
    psd: np.ndarray
    relative_band_power: dict[str, float]
    channel_relative_band_power: dict[str, np.ndarray]
    channel_alpha_power: np.ndarray
    channel_power_1_30: np.ndarray
    channel_valid_mask: np.ndarray
    rms_uv: float
    spectral_entropy: float
    mean_abs_correlation: float
    signal_quality: float
    hjorth_mobility: float
    hjorth_complexity: float
    temporal_activity: float
    spatial_balance: float
    regional_consistency: float


def _as_microvolts(samples: np.ndarray, unit: str) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"EEG samples must be [channels, samples], got {values.shape}")
    if unit.lower() in {"v", "volt", "volts"}:
        return values * 1e6
    return values


def _band_mean(psd: np.ndarray, frequencies: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (frequencies >= low) & (frequencies < high)
    if not np.any(mask):
        return np.zeros(psd.shape[0], dtype=np.float64)
    return np.trapz(psd[:, mask], frequencies[mask], axis=1)


def _welch_psd(samples: np.ndarray, sample_rate: float, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """Small NumPy-only Welch implementation for this display-only feature path."""
    nperseg = min(nperseg, samples.shape[1])
    step = max(1, nperseg // 2)
    starts = list(range(0, samples.shape[1] - nperseg + 1, step)) or [0]
    window = np.hanning(nperseg)
    scale = sample_rate * np.sum(window**2)
    estimates: list[np.ndarray] = []
    for start in starts:
        segment = samples[:, start : start + nperseg]
        if segment.shape[1] < nperseg:
            segment = np.pad(segment, ((0, 0), (0, nperseg - segment.shape[1])))
        spectrum = np.fft.rfft(segment * window, axis=-1)
        estimate = np.abs(spectrum) ** 2 / max(scale, 1e-12)
        if nperseg % 2 == 0:
            estimate[:, 1:-1] *= 2.0
        else:
            estimate[:, 1:] *= 2.0
        estimates.append(estimate)
    return np.fft.rfftfreq(nperseg, d=1.0 / sample_rate), np.mean(estimates, axis=0)


def calculate_signal_quality(
    samples_uv: np.ndarray,
    frequencies: np.ndarray,
    psd: np.ndarray,
    channel_valid_mask: np.ndarray,
) -> float:
    """A pragmatic display-quality score, not an artifact classifier."""
    finite_channel = np.isfinite(samples_uv).all(axis=1) & channel_valid_mask
    if not np.any(finite_channel):
        return 0.0
    valid = samples_uv[finite_channel]
    flat_ratio = float(np.mean(np.std(valid, axis=1) < 0.15))
    clipped_ratio = float(np.mean(np.abs(valid) > 250.0))
    total = np.trapz(psd[finite_channel], frequencies, axis=1) + 1e-12
    high = _band_mean(psd[finite_channel], frequencies, 30.0, 45.0)
    noise_ratio = float(np.mean(high / total))
    # Explicitly invalid channels are excluded from feature quality statistics;
    # their quality is owned by the acquisition/device adapter.
    valid_ratio = float(np.mean(np.isfinite(samples_uv[channel_valid_mask]).all(axis=1))) if np.any(channel_valid_mask) else 0.0
    score = 100.0 * (0.50 * valid_ratio + 0.25 * (1.0 - flat_ratio) + 0.15 * (1.0 - min(1.0, clipped_ratio * 15.0)) + 0.10 * (1.0 - min(1.0, noise_ratio * 4.0)))
    return float(np.clip(score, 0.0, 100.0))


def _hemisphere_balance(total_power: np.ndarray, channel_names: list[str] | None) -> float:
    if not channel_names or len(channel_names) != len(total_power):
        return 0.5
    left: list[float] = []
    right: list[float] = []
    for name, power in zip(channel_names, total_power, strict=True):
        match = re.search(r"(\d+)$", name.upper().replace(" ", ""))
        if match is None:
            continue
        (left if int(match.group(1)) % 2 else right).append(float(power))
    if not left or not right:
        return 0.5
    left_power, right_power = np.mean(left), np.mean(right)
    return float(np.clip(1.0 - abs(left_power - right_power) / (left_power + right_power + 1e-12), 0.0, 1.0))


def _regional_consistency(channel_relative: dict[str, np.ndarray], channel_names: list[str] | None) -> float:
    if not channel_names or not channel_relative:
        return 0.5
    bands = np.vstack([channel_relative[name] for name in EEG_BANDS]).T
    regions: dict[str, list[np.ndarray]] = {"frontal": [], "central": [], "parietal": [], "occipital": []}
    for name, vector in zip(channel_names, bands, strict=True):
        upper = name.upper().replace(" ", "")
        if upper.startswith(("FP", "AF", "F", "FT", "FC")):
            region = "frontal"
        elif upper.startswith(("CP", "P", "PO", "TP")):
            region = "parietal"
        elif upper.startswith(("O", "I")):
            region = "occipital"
        elif upper.startswith(("C", "T")):
            region = "central"
        else:
            continue
        regions[region].append(vector)
    means = [np.mean(values, axis=0) for values in regions.values() if values]
    if len(means) < 2:
        return 0.5
    similarities: list[float] = []
    for index, left in enumerate(means[:-1]):
        for right in means[index + 1 :]:
            similarities.append(float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12)))
    return float(np.clip(np.mean(similarities), 0.0, 1.0)) if similarities else 0.5


def extract_signal_features(
    samples: np.ndarray,
    sample_rate: float,
    *,
    unit: str = "V",
    channel_names: list[str] | None = None,
    channel_valid_mask: np.ndarray | None = None,
) -> SignalFeatures:
    """Return features from valid EEG channels only.

    PSD is retained in original channel order so renderers can map it by name.
    Invalid channels are represented by ``NaN`` in channel-level outputs and
    never participate in aggregate statistics or robust normalization.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    samples_uv = _as_microvolts(samples, unit)
    channels = samples_uv.shape[0]
    if channel_valid_mask is None:
        requested_valid = np.ones(channels, dtype=bool)
    else:
        requested_valid = np.asarray(channel_valid_mask, dtype=bool)
        if requested_valid.ndim != 1 or requested_valid.shape[0] != channels:
            raise ValueError("channel_valid_mask must have shape [channels]")
    finite_channel = np.isfinite(samples_uv).all(axis=1)
    valid_mask = requested_valid & finite_channel
    if not np.any(valid_mask):
        raise ValueError("at least one valid finite EEG channel is required")
    clean = np.nan_to_num(samples_uv, nan=0.0, posinf=0.0, neginf=0.0)
    nperseg = min(clean.shape[1], max(32, int(round(sample_rate * 2))))
    frequencies, psd = _welch_psd(clean - clean.mean(axis=1, keepdims=True), sample_rate, nperseg)
    keep = (frequencies >= 1.0) & (frequencies <= min(45.0, sample_rate / 2.0))
    frequencies, psd = frequencies[keep], psd[:, keep]
    total_by_channel = np.trapz(psd, frequencies, axis=1) + 1e-12
    valid_total_power = total_by_channel[valid_mask]
    relative: dict[str, float] = {}
    channel_relative: dict[str, np.ndarray] = {}
    channel_alpha = np.full(channels, np.nan, dtype=np.float64)
    for name, (low, high) in EEG_BANDS.items():
        power = _band_mean(psd, frequencies, low, high)
        ratio = power / total_by_channel
        ratio[~valid_mask] = np.nan
        relative[name] = float(np.nanmean(ratio[valid_mask]))
        channel_relative[name] = ratio
        if name == "alpha":
            channel_alpha[valid_mask] = power[valid_mask]
    power_1_30 = _band_mean(psd, frequencies, 1.0, 30.0)
    power_1_30[~valid_mask] = np.nan
    band_distribution = np.asarray([relative[name] for name in EEG_BANDS], dtype=np.float64)
    band_distribution = band_distribution / (band_distribution.sum() + 1e-12)
    entropy = -np.sum(band_distribution * np.log(band_distribution + 1e-12)) / np.log(len(band_distribution))
    subset = clean[valid_mask][:32]
    if subset.shape[0] < 2 or subset.shape[1] < 2:
        synchrony = 0.0
    else:
        corr = np.corrcoef(subset)
        upper = np.abs(corr[np.triu_indices_from(corr, k=1)])
        synchrony = float(np.nanmean(upper)) if upper.size and np.isfinite(upper).any() else 0.0
    valid_clean = clean[valid_mask]
    first_difference = np.diff(valid_clean, axis=1)
    second_difference = np.diff(first_difference, axis=1)
    variance = np.var(valid_clean, axis=1) + 1e-12
    first_variance = np.var(first_difference, axis=1) + 1e-12
    second_variance = np.var(second_difference, axis=1) + 1e-12
    mobility_by_channel = np.sqrt(first_variance / variance)
    complexity_by_channel = np.sqrt(second_variance / first_variance) / (mobility_by_channel + 1e-12)
    zero_crossing_rate = np.mean(np.diff(np.signbit(valid_clean), axis=1) != 0)
    temporal_activity = float(np.sqrt(np.mean(first_difference**2)) + np.mean(np.std(valid_clean, axis=1)) * zero_crossing_rate)
    quality = calculate_signal_quality(samples_uv, frequencies, psd, requested_valid)
    valid_names = [name for name, valid in zip(channel_names or [], valid_mask, strict=True) if valid] if channel_names else None
    return SignalFeatures(
        frequencies=frequencies,
        psd=psd,
        relative_band_power=relative,
        channel_relative_band_power=channel_relative,
        channel_alpha_power=channel_alpha,
        channel_power_1_30=power_1_30,
        channel_valid_mask=valid_mask,
        rms_uv=float(np.sqrt(np.mean(valid_clean**2))),
        spectral_entropy=float(np.clip(entropy, 0.0, 1.0)),
        mean_abs_correlation=float(np.clip(synchrony, 0.0, 1.0)),
        signal_quality=quality,
        hjorth_mobility=float(np.nanmedian(mobility_by_channel)),
        hjorth_complexity=float(np.nanmedian(complexity_by_channel)),
        temporal_activity=temporal_activity,
        spatial_balance=_hemisphere_balance(valid_total_power, valid_names),
        regional_consistency=(
            _regional_consistency({name: values[valid_mask] for name, values in channel_relative.items()}, valid_names)
            if valid_names is not None
            else 0.5
        ),
    )
