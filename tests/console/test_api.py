from __future__ import annotations

from collections import Counter

from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "api_version": "v1", "runtime_available": True}


def test_api_responses_do_not_expose_eeg_data_or_absolute_paths() -> None:
    with TestClient(app) as client:
        for endpoint in ("/api/v1/models", "/api/v1/datasets", "/api/v1/system/status"):
            response = client.get(endpoint)
            assert response.status_code == 200
            body = response.text.lower()
            assert '"samples"' not in body
            assert '"raw_eeg"' not in body
            assert '"waveform"' not in body
            assert "e:\\" not in body
            assert "c:\\users" not in body


def test_models_expose_one_formal_live_package_per_model() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/models")
    assert response.status_code == 200
    live_items = [item for item in response.json()["items"] if item["live_verified"]]
    assert Counter(item["model_name"] for item in live_items) == Counter({
        "50M": 1,
        "LaBraM": 1,
        "CBraMod": 1,
    })
    assert all(item["window_sec"] == 4.0 and item["step_sec"] == 0.5 for item in live_items)


def test_live_api_rejects_non_formal_runtime_package() -> None:
    with TestClient(app) as client:
        models = client.get("/api/v1/models").json()["items"]
        non_live = next(item for item in models if item["runtime_verified"] and not item["live_verified"])
        response = client.post("/api/v1/runs/live", json={
            "model_id": non_live["id"],
            "source": "neuracle_jellyfish",
            "compute_device": "cpu",
            "confidence_threshold": 0.55,
        })
    assert response.status_code == 422
