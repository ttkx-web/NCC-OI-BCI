import numpy as np
import pytest

from bci_dayloop.preprocessing.canonical import (
    SignalCanonicalizer,
    normalize_channel_name,
)
from bci_dayloop.runtime.types import RawEEGWindow


def test_tc_is_transposed_to_ct() -> None:
    raw = RawEEGWindow(
        data=np.arange(
            20,
            dtype=np.float32,
        ).reshape(10, 2),
        channel_names=["C3", "C4"],
        sample_rate=100.0,
        unit="uV",
        layout="TC",
    )

    output = SignalCanonicalizer().transform(raw)

    assert output.data.shape == (2, 10)
    assert output.channel_names == ["C3", "C4"]
    assert "transpose:TC->CT" in output.processing_history


def test_v_is_converted_to_uv() -> None:
    raw = RawEEGWindow(
        data=np.array(
            [[1e-6, 2e-6]],
            dtype=np.float32,
        ),
        channel_names=["Cz"],
        sample_rate=100.0,
        unit="V",
    )

    output = SignalCanonicalizer(
        target_unit="uV",
    ).transform(raw)

    np.testing.assert_allclose(
        output.data,
        [[1.0, 2.0]],
        rtol=1e-6,
        atol=1e-6,
    )


def test_mv_is_converted_to_uv() -> None:
    raw = RawEEGWindow(
        data=np.array(
            [[0.001, 0.002]],
            dtype=np.float32,
        ),
        channel_names=["Cz"],
        sample_rate=100.0,
        unit="mV",
    )

    output = SignalCanonicalizer().transform(raw)

    np.testing.assert_allclose(
        output.data,
        [[1.0, 2.0]],
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("FP1", "Fp1"),
        ("FPZ", "Fpz"),
        ("EEG Fp1-Ref", "Fp1"),
        ("CZ.", "Cz"),
        ("T3", "T7"),
        ("T4", "T8"),
    ],
)
def test_channel_name_normalization(
    source: str,
    expected: str,
) -> None:
    assert normalize_channel_name(source) == expected


def test_nan_is_rejected() -> None:
    raw = RawEEGWindow(
        data=np.array(
            [[0.0, np.nan]],
            dtype=np.float32,
        ),
        channel_names=["Cz"],
        sample_rate=100.0,
        unit="uV",
    )

    with pytest.raises(
        ValueError,
        match="NaN or Inf",
    ):
        SignalCanonicalizer().transform(raw)


@pytest.mark.parametrize(
    "sample_rate",
    [0.0, -100.0, float("nan")],
)
def test_invalid_sample_rate_is_rejected(
    sample_rate: float,
) -> None:
    raw = RawEEGWindow(
        data=np.zeros(
            (1, 10),
            dtype=np.float32,
        ),
        channel_names=["Cz"],
        sample_rate=sample_rate,
        unit="uV",
    )

    with pytest.raises(
        ValueError,
        match="sample_rate",
    ):
        SignalCanonicalizer().transform(raw)


def test_metadata_is_preserved() -> None:
    raw = RawEEGWindow(
        data=np.zeros(
            (1, 10),
            dtype=np.float32,
        ),
        channel_names=["Cz"],
        sample_rate=100.0,
        unit="uV",
        trial_id="trial_001",
        window_id="window_001",
        label=2,
        metadata={"session": "1test"},
    )

    output = SignalCanonicalizer().transform(raw)

    assert output.trial_id == "trial_001"
    assert output.window_id == "window_001"
    assert output.label == 2
    assert output.metadata == {"session": "1test"}


def test_channel_count_mismatch_is_rejected() -> None:
    raw = RawEEGWindow(
        data=np.zeros(
            (2, 10),
            dtype=np.float32,
        ),
        channel_names=["Cz"],
        sample_rate=100.0,
        unit="uV",
    )

    with pytest.raises(
        ValueError,
        match="Channel dimension",
    ):
        SignalCanonicalizer().transform(raw)