from __future__ import annotations

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from bci_dayloop.inference.http_service import InferenceServiceRuntime, create_inference_server
from bci_dayloop.inference.multi_head import HeadPrediction, MultiHeadPrediction


class FakePredictor:
    window_seconds = 2.0

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, window: object) -> MultiHeadPrediction:
        self.calls += 1
        assert getattr(window, "data").shape == (2, 3)
        assert getattr(window, "unit") == "uV"
        return MultiHeadPrediction(
            workload=HeadPrediction(1, "high", 0.8, (0.2, 0.8)),
            attention=HeadPrediction(2, "focused", 0.7, (0.1, 0.2, 0.7)),
            emotion=HeadPrediction(0, "negative", 0.6, (0.6, 0.3, 0.1)),
        )


@pytest.fixture
def service() -> tuple[str, FakePredictor]:
    predictor = FakePredictor()
    server = create_inference_server(
        "127.0.0.1", 0, InferenceServiceRuntime(predictor, "fake-package", "cpu")
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", predictor
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "sample_rate_hz": 250,
        "unit": "uV",
        "channel_names": ["C3", "C4"],
        "sequence_start": 7,
        "sequence_end": 9,
        "eeg": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    }


def _post(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(url + "/infer", data=json.dumps(payload).encode(), method="POST")
    with urlopen(request) as response:
        assert response.status == 200
        return json.loads(response.read())


def test_health_and_infer_return_named_predictions(service: tuple[str, FakePredictor]) -> None:
    url, predictor = service
    with urlopen(url + "/health") as response:
        health = json.loads(response.read())
    assert health == {"status": "ok", "model_loaded": True, "model_package": "fake-package", "device": "cpu"}

    response = _post(url, _payload())
    assert predictor.calls == 1
    assert response["sequence_start"] == 7
    assert response["sequence_end"] == 9
    assert response["latency_ms"] >= 0
    assert [item["task_id"] for item in response["predictions"]] == ["workload", "attention", "emotion"]
    assert response["predictions"][0]["probabilities"] == pytest.approx([0.2, 0.8])


def test_bad_schema_returns_4xx_without_calling_predictor(service: tuple[str, FakePredictor]) -> None:
    url, predictor = service
    bad = _payload()
    bad["unit"] = "V"
    with pytest.raises(HTTPError) as error:
        _post(url, bad)
    assert error.value.code == 400
    assert predictor.calls == 0
