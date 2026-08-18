from __future__ import annotations

import asyncio
from urllib.parse import urlparse

from bci_dayloop.serving.backends import ServingBackend
from bci_dayloop.serving.profiles import DeviceProfile
from bci_dayloop.serving.protocol import dumps
from bci_dayloop.serving.session import ClientSession

ALLOWED_PATHS = {"", "/", "/v1/model", "/v1/model/"}


def _connection_path(websocket: object) -> str:
    request = getattr(websocket, "request", None)
    path = getattr(request, "path", None) if request is not None else None
    if not isinstance(path, str) or not path:
        return "/"
    return urlparse(path).path or "/"


async def handle_model_client(
    websocket: object,
    backend: ServingBackend,
    *,
    required_profile: DeviceProfile | None = None,
) -> None:
    path = _connection_path(websocket)
    if path not in ALLOWED_PATHS:
        close = getattr(websocket, "close")
        await close(1008, "unsupported path")
        return
    session = ClientSession(backend, required_profile=required_profile)
    send = getattr(websocket, "send")
    try:
        from websockets.exceptions import ConnectionClosed
    except ImportError:
        ConnectionClosed = Exception  # type: ignore[misc, assignment]
    try:
        await send(dumps(backend.hello_payload()))
        async for message in websocket:  # type: ignore[union-attr]
            replies = await asyncio.to_thread(session.handle_message, message)
            for reply in replies:
                await send(reply)
    except ConnectionClosed:
        return


async def serve_model_service(
    backend: ServingBackend,
    *,
    host: str = "127.0.0.1",
    port: int = 8768,
    required_profile: DeviceProfile | None = None,
    stop: asyncio.Event | None = None,
) -> None:
    try:
        from websockets.asyncio.server import serve
    except ImportError as error:
        raise SystemExit("Missing websockets. pip install websockets") from error

    async def handler(websocket: object) -> None:
        await handle_model_client(
            websocket,
            backend,
            required_profile=required_profile,
        )

    async with serve(handler, host, port, max_size=16 * 1024 * 1024, compression=None):
        print(f"[model-service] ws://{host}:{port}/v1/model")
        print("[model-service] waiting for Passive BCI windows...")
        if stop is None:
            await asyncio.Future()
        else:
            await stop.wait()


def run_model_service(
    backend: ServingBackend,
    *,
    host: str = "127.0.0.1",
    port: int = 8768,
    required_profile: DeviceProfile | None = None,
) -> None:
    asyncio.run(
        serve_model_service(
            backend,
            host=host,
            port=port,
            required_profile=required_profile,
        )
    )
