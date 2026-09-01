from __future__ import annotations

import numpy as np
import pytest

from bci_dayloop.inference.inference_schema import EEGInferenceRequest, InferenceSchemaError


def payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "sample_rate_hz": 250,
        "unit": "uV",
        "channel_names": ["C3", "C4"],
        "sequence_start": 100,
        "sequence_end": 102,
        "eeg": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    }


def test_valid_request_preserves_ct_layout() -> None:
    request = EEGInferenceRequest.from_payload(payload())
    assert request.eeg.shape == (2, 3)
    assert request.eeg.dtype == np.float32
    assert request.channel_names == ("C3", "C4")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("unit", "V", "unit"),
        ("sample_rate_hz", 0, "sample_rate_hz"),
        ("channel_names", ["C3"], "channel count"),
        ("sequence_end", 99, "sequence_end"),
        ("eeg", [[0.1, float("nan"), 0.3], [0.2, 0.3, 0.4]], "NaN"),
        ("eeg", [[0.1, float("inf"), 0.3], [0.2, 0.3, 0.4]], "NaN"),
    ],
)
def test_invalid_contract_values_are_rejected(field: str, value: object, message: str) -> None:
    request = payload()
    request[field] = value
    with pytest.raises(InferenceSchemaError, match=message):
        EEGInferenceRequest.from_payload(request)


def test_inconsistent_sequence_length_is_rejected() -> None:
    request = payload()
    request["sequence_end"] = 103
    with pytest.raises(InferenceSchemaError, match="range length"):
        EEGInferenceRequest.from_payload(request)
