from __future__ import annotations

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

