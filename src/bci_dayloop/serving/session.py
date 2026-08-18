from __future__ import annotations

from typing import Any

from bci_dayloop.serving.backends import ServingBackend
from bci_dayloop.serving.profiles import DeviceProfile
from bci_dayloop.serving.protocol import (
    ProtocolError,
    complete_window,
    dumps,
    error_message,
    parse_feedback,
    parse_json_object,
    parse_window_header,
    prediction_message,
)


class ClientSession:
    """Per-connection window/feedback state for one Passive BCI client."""

    def __init__(
        self,
        backend: ServingBackend,
        *,
        required_profile: DeviceProfile | None = None,
    ) -> None:
        self.backend = backend
        self.required_profile = required_profile
        self._pending_header: dict[str, Any] | None = None
        self._seen_request_ids: set[str] = set()
        self._seen_feedback_ids: set[str] = set()
        self._segment_id: str | None = None
        self._last_window_id: int | None = None

    def handle_message(self, message: str | bytes) -> list[str]:
        try:
            if isinstance(message, (bytes, bytearray, memoryview)):
                return [self._complete_binary(bytes(message))]
            if not isinstance(message, str):
                raise ProtocolError("Unsupported WebSocket frame type", code="invalid_json")
            payload = parse_json_object(message)
            message_type = payload.get("type")
            if message_type == "hello":
                return [dumps(self.backend.hello_payload())]
            if message_type == "window":
                if self._pending_header is not None:
                    raise ProtocolError(
                        "window payload was expected before another header",
                        code="invalid_window",
                        request_id=self._pending_header.get("request_id")
                        if isinstance(self._pending_header.get("request_id"), str)
                        else None,
                    )
                self._pending_header = parse_window_header(payload)
                return []
            if message_type == "feedback":
                return [self._handle_feedback(payload)]
            raise ProtocolError(
                f"Unknown client message type: {message_type!r}",
                code="invalid_json",
            )
        except ProtocolError as error:
            self._pending_header = None
            return [dumps(error_message(error))]
        except Exception as error:
            request_id = None
            if self._pending_header and isinstance(self._pending_header.get("request_id"), str):
                request_id = str(self._pending_header["request_id"])
            self._pending_header = None
            return [dumps(error_message(error, request_id=request_id))]

    def _complete_binary(self, payload: bytes) -> str:
        header = self._pending_header
        if header is None:
            raise ProtocolError("binary window payload arrived without a header", code="invalid_window")
        self._pending_header = None
        window = complete_window(header, payload, required_profile=self.required_profile)
        if window.request_id in self._seen_request_ids:
            raise ProtocolError(
                "duplicate request_id",
                code="duplicate_request_id",
                request_id=window.request_id,
            )
        if self._segment_id != window.segment_id:
            self._segment_id = window.segment_id
            self._last_window_id = None
        if self._last_window_id is not None:
            if window.window_id == self._last_window_id:
                raise ProtocolError(
                    "duplicate window_id",
                    code="duplicate_window_id",
                    request_id=window.request_id,
                )
            if window.window_id < self._last_window_id:
                raise ProtocolError(
                    "window_id is not monotonic",
                    code="non_monotonic_window",
                    request_id=window.request_id,
                )
        self._seen_request_ids.add(window.request_id)
        self._last_window_id = window.window_id
        result = self.backend.predict(window)
        return dumps(prediction_message(result))

    def _handle_feedback(self, payload: dict[str, Any]) -> str:
        feedback = parse_feedback(payload)
        duplicate = feedback.feedback_id in self._seen_feedback_ids
        self._seen_feedback_ids.add(feedback.feedback_id)
        ack = self.backend.ack_feedback(feedback)
        ack["duplicate"] = duplicate
        if duplicate:
            ack["accepted"] = False
            ack["reason"] = "duplicate feedback_id"
        return dumps(ack)
