from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bci_dayloop.serving.backends import MockModelBackend
from bci_dayloop.serving.profiles import BCIGO32_PROFILE, NEURACLE59_PROFILE
from bci_dayloop.serving.protocol import (
    ProtocolError,
    complete_window,
    parse_feedback,
    parse_window_header,
    prediction_message,
)
from bci_dayloop.serving.session import ClientSession


def _header(profile, *, window_id=1, request_id="window-1", segment_id="seg-1") -> dict[str, object]:
    return {
        "type": "window",
        "schema_version": 1,
        "request_id": request_id,
        "window_id": window_id,
        "segment_id": segment_id,
        "sample_rate": profile.sample_rate,
        "channel_names": list(profile.channel_names),
        "unit": "uV",
        "layout": "CT",
        "channels": profile.channels,
        "samples": profile.samples,
        "sample_count": profile.samples,
        "start_time_sec": 0.0,
        "end_time_sec": profile.window_sec,
    }


def _payload(profile) -> bytes:
    return np.zeros((profile.channels, profile.samples), dtype="<f4").tobytes()


def test_window_header_and_payload_roundtrip_neuracle59():
    header = parse_window_header(_header(NEURACLE59_PROFILE))
    window = complete_window(header, _payload(NEURACLE59_PROFILE), required_profile=NEURACLE59_PROFILE)
    assert window.profile is not None
    assert window.profile.id == "neuracle59"
    assert window.data.shape == (59, 4000)


def test_window_rejects_non_uv_and_wrong_layout():
    header = _header(NEURACLE59_PROFILE)
    header["unit"] = "V"
    with pytest.raises(ProtocolError, match="uV"):
        parse_window_header(header)
    header = _header(BCIGO32_PROFILE)
    header["layout"] = "TC"
    with pytest.raises(ProtocolError, match="CT"):
        parse_window_header(header)


def test_mock_session_predicts_fixed_probabilities():
    backend = MockModelBackend.from_profile_id("neuracle59")
    session = ClientSession(backend, required_profile=NEURACLE59_PROFILE)
    hello = json.loads(session.handle_message(json.dumps({"type": "hello"}))[0])
    assert hello["service"] == "ncc-dev-mock"
    assert hello["input"]["window_sec"] == 4.0
    assert hello["input"]["step_sec"] == 0.5
    assert session.handle_message(json.dumps(_header(NEURACLE59_PROFILE))) == []
    replies = session.handle_message(_payload(NEURACLE59_PROFILE))
    prediction = json.loads(replies[0])
    assert prediction["type"] == "prediction"
    assert prediction["class_name"] == "left_hand"
    assert prediction["probabilities"] == [0.55, 0.20, 0.15, 0.10]
    assert prediction["confidence"] == 0.55
    assert "output_semantics" not in prediction


def test_session_rejects_duplicate_and_non_monotonic_window_ids():
    backend = MockModelBackend.from_profile_id("bcigo32")
    session = ClientSession(backend, required_profile=BCIGO32_PROFILE)
    session.handle_message(json.dumps(_header(BCIGO32_PROFILE, window_id=1, request_id="a")))
    first = json.loads(session.handle_message(_payload(BCIGO32_PROFILE))[0])
    assert first["type"] == "prediction"

    session.handle_message(json.dumps(_header(BCIGO32_PROFILE, window_id=1, request_id="b")))
    duplicate = json.loads(session.handle_message(_payload(BCIGO32_PROFILE))[0])
    assert duplicate["code"] == "duplicate_window_id"

    session.handle_message(json.dumps(_header(BCIGO32_PROFILE, window_id=3, request_id="c")))
    json.loads(session.handle_message(_payload(BCIGO32_PROFILE))[0])
    session.handle_message(json.dumps(_header(BCIGO32_PROFILE, window_id=2, request_id="d")))
    older = json.loads(session.handle_message(_payload(BCIGO32_PROFILE))[0])
    assert older["code"] == "non_monotonic_window"


def test_feedback_ack_and_duplicate_feedback_id():
    backend = MockModelBackend.from_profile_id("neuracle59")
    session = ClientSession(backend, required_profile=NEURACLE59_PROFILE)
    payload = {
        "type": "feedback",
        "schema_version": 1,
        "feedback_id": "feedback-1",
        "observation_id": "obs-1",
        "label": 1,
        "timestamp_sec": 1.0,
    }
    ack = json.loads(session.handle_message(json.dumps(payload))[0])
    assert ack["type"] == "feedback_ack"
    assert ack["accepted"] is True
    assert ack["duplicate"] is False
    duplicate = json.loads(session.handle_message(json.dumps(payload))[0])
    assert duplicate["accepted"] is False
    assert duplicate["duplicate"] is True
    parsed = parse_feedback(payload)
    assert parsed.label == 1


def test_prediction_message_omits_reward_semantics_for_mi_head():
    backend = MockModelBackend.from_profile_id("neuracle59")
    session = ClientSession(backend, required_profile=NEURACLE59_PROFILE)
    session.handle_message(json.dumps(_header(NEURACLE59_PROFILE)))
    prediction = json.loads(session.handle_message(_payload(NEURACLE59_PROFILE))[0])
    encoded = prediction_message(
        backend.predict(
            complete_window(
                parse_window_header(_header(NEURACLE59_PROFILE, request_id="window-2", window_id=2)),
                _payload(NEURACLE59_PROFILE),
                required_profile=NEURACLE59_PROFILE,
            )
        )
    )
    assert "output_semantics" not in encoded
    assert prediction["class_names"] == ["left_hand", "right_hand", "feet", "tongue"]


def test_two_second_window_keeps_device_identity():
    from bci_dayloop.serving.profiles import profile_with_window

    two_sec = profile_with_window(BCIGO32_PROFILE, 2.0)
    assert two_sec.samples == 500
    header = parse_window_header(_header(two_sec))
    window = complete_window(header, _payload(two_sec))
    assert window.profile is not None
    assert window.profile.id == "bcigo32"
    assert window.samples == 500

    matched = complete_window(header, _payload(two_sec), required_profile=two_sec)
    assert matched.profile is not None
    assert matched.profile.id == "bcigo32"
    assert matched.samples == 500


def test_required_profile_explains_layout_mismatch():
    header = parse_window_header(_header(BCIGO32_PROFILE))
    with pytest.raises(ProtocolError, match=r"neuracle59.*32ch @ 250 Hz"):
        complete_window(header, _payload(BCIGO32_PROFILE), required_profile=NEURACLE59_PROFILE)


def test_mock_accepts_bcigo_without_required_profile():
    backend = MockModelBackend.from_profile_id("neuracle59")
    session = ClientSession(backend)
    session.handle_message(json.dumps(_header(BCIGO32_PROFILE)))
    prediction = json.loads(session.handle_message(_payload(BCIGO32_PROFILE))[0])
    assert prediction["type"] == "prediction"
    assert prediction["class_name"] == "left_hand"
