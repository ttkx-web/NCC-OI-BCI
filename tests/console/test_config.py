from __future__ import annotations

import pytest

from app.config import Settings


def test_jellyfish_endpoint_uses_safe_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEURACLE_JELLYFISH_HOST", raising=False)
    monkeypatch.delenv("NEURACLE_JELLYFISH_PORT", raising=False)

    settings = Settings()

    assert settings.neuracle_jellyfish_host == "127.0.0.1"
    assert settings.neuracle_jellyfish_port == 8712


def test_jellyfish_endpoint_uses_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEURACLE_JELLYFISH_HOST", "jellyfish.internal")
    monkeypatch.setenv("NEURACLE_JELLYFISH_PORT", "18712")

    settings = Settings()

    assert settings.neuracle_jellyfish_host == "jellyfish.internal"
    assert settings.neuracle_jellyfish_port == 18712


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("NEURACLE_JELLYFISH_HOST", "   "),
        ("NEURACLE_JELLYFISH_PORT", "not-a-port"),
        ("NEURACLE_JELLYFISH_PORT", "0"),
        ("NEURACLE_JELLYFISH_PORT", "65536"),
    ],
)
def test_invalid_jellyfish_endpoint_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.delenv("NEURACLE_JELLYFISH_HOST", raising=False)
    monkeypatch.delenv("NEURACLE_JELLYFISH_PORT", raising=False)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError):
        Settings()
