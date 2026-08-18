from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bci_dayloop.models.cbramod.config import (
    CBraModConfig,
)
from bci_dayloop.models.cbramod.preprocessing import (
    CBraModPipelinePreprocessor,
)
from bci_dayloop.runtime.types import CanonicalEEGWindow
from bci_dayloop.runtime.types import RawEEGWindow
from bci_dayloop.preprocessing.canonical import SignalCanonicalizer

from bci_dayloop.models.cbramod.config import (
    BCICIV2A_22_CHANNELS,
    CBraModConfig,
)


def build_preprocessor(
    *,
    missing_channel_policy: str,
    normalization: str = "none",
    window_seconds: float = 4.0,
    time_segments: int = 4,
    min_observed_channels: int | None = None,
) -> CBraModPipelinePreprocessor:
    target_channels = BCICIV2A_22_CHANNELS

    if min_observed_channels is None:
        min_observed_channels = (
            len(target_channels) if missing_channel_policy == "error" else 2
        )

    config = CBraModConfig(
        checkpoint_path=Path("unused_backbone.pt"),
        device="cpu",
        filter_enabled=False,
        reference_mode="none",
        normalization=normalization,
        window_seconds=window_seconds,
        time_segments=time_segments,
        missing_channel_policy=missing_channel_policy,
        min_observed_channels=min_observed_channels,
        spline_alpha=1e-5,
    )

    return CBraModPipelinePreprocessor(config)


def test_fixed_100uv_exact_values_and_shape() -> None:
    preprocessor = build_preprocessor(
        missing_channel_policy="error", normalization="fixed_100uv",
        window_seconds=2.0, time_segments=2,
    )
    signal = np.zeros((22, 400), dtype=np.float32)
    signal[:, 0], signal[:, 1], signal[:, 2] = 100.0, 50.0, -100.0
    actual = prepared_signal(preprocessor, make_window(
        signal=signal, channel_names=list(BCICIV2A_22_CHANNELS)
    ))
    assert actual.shape == (1, 22, 2, 200)
    assert actual.dtype == np.float32 and np.isfinite(actual).all()
    np.testing.assert_allclose(actual[0, :, 0, :3], [
        [1.0, 0.5, -1.0]
    ] * 22, rtol=0.0, atol=1e-7)


def test_volts_canonicalize_to_uv_then_fixed_100uv() -> None:
    raw = RawEEGWindow(
        data=np.full((22, 400), 100e-6, dtype=np.float32),
        channel_names=list(BCICIV2A_22_CHANNELS), sample_rate=200.0,
        unit="V", layout="CT", window_id="v-to-uv",
    )
    canonical = SignalCanonicalizer(target_unit="uV").transform(raw)
    np.testing.assert_allclose(canonical.data, 100.0, rtol=0.0, atol=1e-4)
    preprocessor = build_preprocessor(
        missing_channel_policy="error", normalization="fixed_100uv",
        window_seconds=2.0, time_segments=2,
    )
    actual = prepared_signal(preprocessor, canonical)
    np.testing.assert_allclose(actual, 1.0, rtol=0.0, atol=1e-6)


def test_workload_250hz_spline_two_second_shape_and_diagnostics() -> None:
    preprocessor = build_preprocessor(
        missing_channel_policy="spherical_spline", normalization="fixed_100uv",
        window_seconds=2.0, time_segments=2, min_observed_channels=21,
    )
    observed = [name for name in BCICIV2A_22_CHANNELS if name != "Cz"]
    time = np.arange(500, dtype=np.float32) / 250.0
    signal = np.stack([np.sin(2 * np.pi * (i + 1) * time) for i in range(21)]).astype(np.float32)
    window = CanonicalEEGWindow(
        data=signal, channel_names=observed, sample_rate=250.0,
        unit="uV", trial_id="workload", window_id="workload-2s",
    )
    actual = prepared_signal(preprocessor, window)
    assert actual.shape == (1, 22, 2, 200) and np.isfinite(actual).all()
    diagnostics = preprocessor.last_diagnostics
    assert diagnostics is not None
    assert [name.upper() for name in diagnostics.missing_channel_names] == ["CZ"]
    assert diagnostics.completion_policy == "spherical_spline"
    assert diagnostics.normalization == "fixed_100uv"


def test_none_and_per_window_zscore_remain_compatible() -> None:
    names = list(BCICIV2A_22_CHANNELS)
    signal = make_signal(22, 400)
    none_values = prepared_signal(build_preprocessor(
        missing_channel_policy="error", normalization="none",
        window_seconds=2.0, time_segments=2,
    ), make_window(signal=signal, channel_names=names))
    np.testing.assert_allclose(none_values.reshape(22, 400), signal, rtol=0.0, atol=1e-6)
    zscore_values = prepared_signal(build_preprocessor(
        missing_channel_policy="error", normalization="per_window_zscore",
        window_seconds=2.0, time_segments=2,
    ), make_window(signal=signal, channel_names=names)).reshape(22, 400)
    np.testing.assert_allclose(zscore_values.mean(axis=-1), 0.0, atol=1e-6)
    np.testing.assert_allclose(zscore_values.std(axis=-1), 1.0, atol=1e-5)
    assert CBraModConfig(checkpoint_path="unused.pt").missing_channel_policy == "error"


def make_signal(
    n_channels: int,
    n_samples: int,
) -> np.ndarray:
    time = np.linspace(
        0.0,
        4.0,
        n_samples,
        endpoint=False,
        dtype=np.float32,
    )

    return np.stack(
        [
            (
                (index + 1) * np.sin(
                    2.0 * np.pi * (index + 1) * time
                )
                + 0.01 * index
            )
            for index in range(n_channels)
        ],
        axis=0,
    ).astype(np.float32)


def make_window(
    *,
    signal: np.ndarray,
    channel_names: list[str],
) -> CanonicalEEGWindow:
    return CanonicalEEGWindow(
        data=signal,
        channel_names=channel_names,
        sample_rate=200.0,
        unit="uV",
        trial_id="test_trial",
        window_id="test_window",
    )


def prepared_signal(
    preprocessor: CBraModPipelinePreprocessor,
    window: CanonicalEEGWindow,
) -> np.ndarray:
    prepared = preprocessor.transform(window)
    return (
        prepared.model_input["signal"]
        .detach()
        .cpu()
        .numpy()
        .copy()
    )


def test_complete_channels_are_permutation_invariant() -> None:
    preprocessor = build_preprocessor(
        missing_channel_policy="error"
    )

    target_names = list(
        preprocessor.config.standard_channels
    )
    signal = make_signal(
        len(target_names),
        preprocessor.config.num_samples,
    )

    expected = prepared_signal(
        preprocessor,
        make_window(
            signal=signal,
            channel_names=target_names,
        ),
    )

    permutation = np.arange(len(target_names))
    np.random.default_rng(42).shuffle(permutation)

    actual = prepared_signal(
        preprocessor,
        make_window(
            signal=signal[permutation],
            channel_names=[
                target_names[index]
                for index in permutation
            ],
        ),
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-6,
    )


def test_unknown_channels_are_dropped() -> None:
    preprocessor = build_preprocessor(
        missing_channel_policy="error"
    )

    target_names = list(
        preprocessor.config.standard_channels
    )
    signal = make_signal(
        len(target_names),
        preprocessor.config.num_samples,
    )

    expected = prepared_signal(
        preprocessor,
        make_window(
            signal=signal,
            channel_names=target_names,
        ),
    )

    auxiliary = np.full(
        (1, preprocessor.config.num_samples),
        12345.0,
        dtype=np.float32,
    )

    actual = prepared_signal(
        preprocessor,
        make_window(
            signal=np.concatenate(
                [signal, auxiliary],
                axis=0,
            ),
            channel_names=target_names + ["AUX-1"],
        ),
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-6,
    )

    diagnostics = preprocessor.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.unknown_channel_names == ("AUX-1",)


def test_duplicate_channels_are_averaged() -> None:
    preprocessor = build_preprocessor(
        missing_channel_policy="error"
    )

    target_names = list(
        preprocessor.config.standard_channels
    )
    base_signal = make_signal(
        len(target_names),
        preprocessor.config.num_samples,
    )

    first_a = base_signal[0] - 2.0
    first_b = base_signal[0] + 2.0

    duplicate_signal = np.concatenate(
        [
            first_a[None, :],
            first_b[None, :],
            base_signal[1:],
        ],
        axis=0,
    )

    duplicate_names = [
        target_names[0],
        target_names[0],
        *target_names[1:],
    ]

    actual = prepared_signal(
        preprocessor,
        make_window(
            signal=duplicate_signal,
            channel_names=duplicate_names,
        ),
    )

    # 必须在下一次 transform 前读取。
    duplicate_diagnostics = preprocessor.last_diagnostics
    assert duplicate_diagnostics is not None
    assert duplicate_diagnostics.duplicate_channel_count == 1

    expected = prepared_signal(
        preprocessor,
        make_window(
            signal=base_signal,
            channel_names=target_names,
        ),
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-6,
    )


def test_strict_mode_rejects_missing_channels() -> None:
    preprocessor = build_preprocessor(
        missing_channel_policy="error"
    )

    target_names = list(
        preprocessor.config.standard_channels
    )
    signal = make_signal(
        len(target_names),
        preprocessor.config.num_samples,
    )

    with pytest.raises(
        ValueError,
        match="Missing channels",
    ):
        preprocessor.transform(
            make_window(
                signal=signal[:-1],
                channel_names=target_names[:-1],
            )
        )


def test_spherical_spline_completion_is_finite_and_cached() -> None:
    preprocessor = build_preprocessor(
        missing_channel_policy="spherical_spline"
    )

    target_names = list(
        preprocessor.config.standard_channels
    )
    signal = make_signal(
        len(target_names),
        preprocessor.config.num_samples,
    )

    missing_indices = set(
        range(0, len(target_names), 7)
    )
    observed_indices = [
        index
        for index in range(len(target_names))
        if index not in missing_indices
    ]

    window = make_window(
        signal=signal[observed_indices],
        channel_names=[
            target_names[index]
            for index in observed_indices
        ],
    )

    first = prepared_signal(preprocessor, window)

    diagnostics = preprocessor.last_diagnostics
    assert diagnostics is not None
    assert len(diagnostics.missing_channel_names) == len(
        missing_indices
    )
    assert diagnostics.completion_policy == "spherical_spline"
    assert diagnostics.completion_matrix_sha256 is not None
    assert np.isfinite(first).all()

    for index in missing_indices:
        assert np.any(np.abs(first[0, index]) > 1e-7)

    first_sha256 = diagnostics.completion_matrix_sha256
    second = prepared_signal(preprocessor, window)

    diagnostics = preprocessor.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.completion_matrix_sha256 == first_sha256
    assert len(preprocessor._completion_matrix_cache) == 1

    np.testing.assert_allclose(
        first,
        second,
        rtol=0.0,
        atol=1e-6,
    )
