"""Dependency-free localhost HTTP wrapper for a loaded raw-window predictor."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from bci_dayloop.inference.inference_schema import (
    EEGInferenceRequest,
    EEGInferenceResponse,
    InferenceSchemaError,
    SCHEMA_VERSION,
)
from bci_dayloop.inference.predictor import RawWindowPredictor
from bci_dayloop.inference.window_inference import infer_eeg_window, named_predictions


@dataclass(frozen=True, slots=True)
class InferenceServiceRuntime:
    """The already-loaded runtime shared by every request to one server."""

    predictor: RawWindowPredictor
    model_package: str
    device: str


def create_inference_server(
    host: str,
    port: int,
    runtime: InferenceServiceRuntime,
) -> HTTPServer:
    """Create a server; the caller owns its lifecycle and loaded runtime."""

    class InferenceHandler(BaseHTTPRequestHandler):
        server_version = "NCCOIInference/1.0"

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: HTTPStatus, message: str) -> None:
            self._send_json(status, {"error": message})

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._error(HTTPStatus.NOT_FOUND, "Not found.")
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model_loaded": True,
                    "model_package": runtime.model_package,
                    "device": runtime.device,
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/infer":
                self._error(HTTPStatus.NOT_FOUND, "Not found.")
                return
            try:
                content_length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._error(HTTPStatus.BAD_REQUEST, "Content-Length must be an integer.")
                return
            if content_length < 0 or content_length > 64 * 1024 * 1024:
                self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request body exceeds 64 MiB.")
                return
            try:
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                request = EEGInferenceRequest.from_payload(payload)
            except (UnicodeDecodeError, json.JSONDecodeError, InferenceSchemaError) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
                return

            try:
                # latency_ms intentionally starts after request parsing and excludes HTTP/JSON work.
                started = time.perf_counter()
                result = infer_eeg_window(
                    runtime.predictor,
                    eeg=request.eeg,
                    sample_rate_hz=request.sample_rate_hz,
                    channel_names=request.channel_names,
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                response = EEGInferenceResponse(
                    schema_version=SCHEMA_VERSION,
                    sequence_start=request.sequence_start,
                    sequence_end=request.sequence_end,
                    predictions=named_predictions(result),
                    latency_ms=latency_ms,
                )
            except ValueError as error:
                self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(error))
                return
            except Exception:  # Preserve details in the server log, never send a traceback to a client.
                logging.exception("Unhandled inference failure")
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "Inference failed.")
                return
            self._send_json(HTTPStatus.OK, response.to_payload())

        def log_message(self, format: str, *args: object) -> None:
            logging.info("%s - %s", self.address_string(), format % args)

    return HTTPServer((host, port), InferenceHandler)
