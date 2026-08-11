from __future__ import annotations

import asyncio
import queue

from fastapi import WebSocket, WebSocketDisconnect

from app.schemas.events import error_event
from app.schemas.runs import RunState
from app.services.run_service import TERMINAL_STATES, RunService


async def stream_run(websocket: WebSocket, run_id: str, service: RunService) -> None:
    await websocket.accept()
    try:
        record = service.get(run_id)
    except LookupError:
        await websocket.send_json(
            error_event(run_id, code="RUN_NOT_FOUND", message="运行记录不存在。", fatal=True)
        )
        await websocket.close(code=4404)
        return

    subscriber = record.broker.subscribe()
    try:
        while True:
            try:
                event = await asyncio.to_thread(subscriber.get, True, 0.5)
            except queue.Empty:
                if record.state in TERMINAL_STATES and subscriber.empty():
                    await websocket.close(code=1000)
                    return
                continue
            await websocket.send_json(event)
    except WebSocketDisconnect:
        return
    finally:
        record.broker.unsubscribe(subscriber)

