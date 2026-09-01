from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import verify_three_state_inference as offline  # noqa: E402
from bci_dayloop.inference.inference_schema import EEGInferenceRequest, Prediction


def test_fixture_export_preserves_contract_window_and_direct_predictions(tmp_path: Path) -> None:
    request = EEGInferenceRequest.from_payload({
        "schema_version": "1.0",
        "sample_rate_hz": 250.0,
        "unit": "uV",
        "channel_names": ["C3", "C4"],
        "sequence_start": 0,
        "sequence_end": 2,
        "eeg": np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32).tolist(),
    })
    direct = (
        Prediction(
            task_id="example_task",
            class_id=np.int64(1),
            label="active",
            confidence=np.float32(0.75),
            probabilities=(np.float32(0.25), np.float32(0.75)),
        ),
    )

    request_path = offline.write_fixture(tmp_path / "nested" / "request.json", offline.request_fixture_payload(request))
    reference_path = offline.write_fixture(
        tmp_path / "nested" / "reference.json",
        offline.reference_fixture_payload(request, direct, latency_ms=np.float32(12.5)),
    )
    request_json = json.loads(request_path.read_text(encoding="utf-8"))
    reference_json = json.loads(reference_path.read_text(encoding="utf-8"))

    assert EEGInferenceRequest.from_payload(request_json).eeg.shape == (2, 3)
    assert request_json["sample_rate_hz"] == 250
    assert isinstance(request_json["sample_rate_hz"], int)
    assert request_json["sequence_start"] == reference_json["sequence_start"] == 0
    assert request_json["sequence_end"] == reference_json["sequence_end"] == 2
    np.testing.assert_allclose(request_json["eeg"], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    prediction = reference_json["predictions"][0]
    assert prediction["class_id"] == 1
    assert prediction["confidence"] == pytest.approx(0.75)
    assert prediction["probabilities"] == pytest.approx([0.25, 0.75])


def test_explicit_channel_subset_preserves_requested_order() -> None:
    eeg = np.arange(15, dtype=np.float32).reshape(3, 5)
    selected, names = offline.select_input_channels(eeg, ["Fz", "C3", "C4"], "C4,Fz")
    assert names == ["C4", "Fz"]
    np.testing.assert_array_equal(selected, eeg[[2, 0]])


def test_verify_cli_exposes_all_modes_and_fixture_options() -> None:
    parser = offline.build_parser()
    assert parser.parse_args(["--mode", "direct"]).mode == "direct"
    assert parser.parse_args(["--mode", "package"]).mode == "package"
    assert parser.parse_args(["--mode", "decoder"]).mode == "decoder"
    args = parser.parse_args(["--mode", "http", "--server-url", "http://127.0.0.1:8767"])
    assert args.server_url == "http://127.0.0.1:8767"
    assert parser.parse_args(["--mode", "all", "--export-request", "request.json"]).mode == "all"
