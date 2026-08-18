from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bci_dayloop.serving.backends import MockModelBackend
from bci_dayloop.serving.profiles import NEURACLE59_PROFILE
from bci_dayloop.serving.server import handle_model_client


def _header() -> dict[str, object]:
    profile = NEURACLE59_PROFILE
    return {
        "type": "window",
        "schema_version": 1,
        "request_id": "window-live-1",
        "window_id": 1,
        "segment_id": "neuracle-seg-1",
        "sample_rate": profile.sample_rate,
        "channel_names": list(profile.channel_names),
        "unit": "uV",
        "layout": "CT",
        "channels": profile.channels,
        "samples": profile.samples,
        "start_time_sec": 0.0,
        "end_time_sec": 4.0,
    }


async def _roundtrip() -> None:
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    backend = MockModelBackend.from_profile_id("neuracle59")

    async def handler(websocket: object) -> None:
        await handle_model_client(
            websocket,
            backend,
            required_profile=NEURACLE59_PROFILE,
        )

    async with serve(handler, "127.0.0.1", 0, max_size=16 * 1024 * 1024) as server:
        port = int(server.sockets[0].getsockname()[1])
        async with connect(f"ws://127.0.0.1:{port}/v1/model", max_size=16 * 1024 * 1024) as client:
            hello = json.loads(await client.recv())
            assert hello["type"] == "hello"
            assert hello["service"] == "ncc-dev-mock"
            await client.send(json.dumps({"type": "hello", "schema_version": 1, "client": "passive-bci"}))
            hello_reply = json.loads(await client.recv())
            assert hello_reply["model"]["class_names"] == [
                "left_hand",
                "right_hand",
                "feet",
                "tongue",
            ]
            await client.send(json.dumps(_header()))
            await client.send(np.zeros((59, 4000), dtype="<f4").tobytes())
            prediction = json.loads(await client.recv())
            assert prediction["type"] == "prediction"
            assert prediction["class_id"] == 0
            assert prediction["window_id"] == 1


def test_mock_websocket_accepts_passive_bci_window_frames():
    asyncio.run(_roundtrip())


async def _abrupt_disconnect() -> None:
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    backend = MockModelBackend.from_profile_id("neuracle59")
    errors: list[BaseException] = []

    async def handler(websocket: object) -> None:
        try:
            await handle_model_client(
                websocket,
                backend,
                required_profile=NEURACLE59_PROFILE,
            )
        except BaseException as exc:
            errors.append(exc)
            raise

    async with serve(handler, "127.0.0.1", 0, max_size=16 * 1024 * 1024) as server:
        port = int(server.sockets[0].getsockname()[1])
        async with connect(f"ws://127.0.0.1:{port}/v1/model") as client:
            hello = json.loads(await client.recv())
            assert hello["type"] == "hello"
            transport = getattr(client, "transport", None)
            if transport is not None:
                transport.abort()
        await asyncio.sleep(0.05)
    assert errors == []


def test_abrupt_client_disconnect_is_not_a_handler_failure():
    asyncio.run(_abrupt_disconnect())
