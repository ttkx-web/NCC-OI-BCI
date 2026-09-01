"""Schema, one-window entry point, and localhost HTTP service."""
from __future__ import annotations
import json, logging, time
from dataclasses import dataclass, fields
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Mapping, Protocol, Sequence
import numpy as np
from bci_dayloop.applications.three_mental_states.contract import HeadPrediction, ThreeMentalStatePrediction
from bci_dayloop.runtime.types import RawEEGWindow
SCHEMA_VERSION = "1.0"
class RawWindowPredictor(Protocol):
    def predict(self, window: RawEEGWindow) -> ThreeMentalStatePrediction: ...
class InferenceSchemaError(ValueError): pass
def _required(payload: Mapping[str, Any], field: str) -> Any:
    if field not in payload: raise InferenceSchemaError(f"Missing required field: {field}.")
    return payload[field]
@dataclass(frozen=True, slots=True)
class EEGInferenceRequest:
    schema_version: str; sample_rate_hz: float; unit: str; channel_names: tuple[str, ...]; sequence_start: int; sequence_end: int; eeg: np.ndarray
    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EEGInferenceRequest":
        if not isinstance(payload, Mapping): raise InferenceSchemaError("Request JSON must be an object.")
        if _required(payload, "schema_version") != SCHEMA_VERSION: raise InferenceSchemaError(f"schema_version must be {SCHEMA_VERSION!r}, got {payload.get('schema_version')!r}.")
        if _required(payload, "unit") != "uV": raise InferenceSchemaError("unit must be exactly 'uV'.")
        sample_rate = _required(payload, "sample_rate_hz")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, float)): raise InferenceSchemaError("sample_rate_hz must be a number.")
        sample_rate = float(sample_rate)
        if not np.isfinite(sample_rate) or sample_rate <= 0: raise InferenceSchemaError("sample_rate_hz must be finite and greater than zero.")
        raw_names = _required(payload, "channel_names")
        if not isinstance(raw_names, list) or not raw_names or any(not isinstance(name, str) or not name.strip() for name in raw_names): raise InferenceSchemaError("channel_names must be a non-empty array containing non-empty strings.")
        start, end = _required(payload, "sequence_start"), _required(payload, "sequence_end")
        if isinstance(start, bool) or not isinstance(start, int): raise InferenceSchemaError("sequence_start must be an integer.")
        if isinstance(end, bool) or not isinstance(end, int): raise InferenceSchemaError("sequence_end must be an integer.")
        if end < start: raise InferenceSchemaError("sequence_end must be greater than or equal to sequence_start.")
        if not isinstance(_required(payload, "eeg"), list): raise InferenceSchemaError("eeg must be a two-dimensional JSON array in [C, T] layout.")
        try: eeg = np.asarray(payload["eeg"], dtype=np.float32)
        except (TypeError, ValueError) as error: raise InferenceSchemaError("eeg must contain numeric values only.") from error
        if eeg.ndim != 2: raise InferenceSchemaError(f"eeg must have shape [C, T], got {eeg.shape}.")
        if eeg.shape[0] != len(raw_names): raise InferenceSchemaError(f"eeg channel count must equal len(channel_names): {eeg.shape[0]} != {len(raw_names)}.")
        if eeg.shape[1] <= 0: raise InferenceSchemaError("eeg must contain at least one sample per channel.")
        if end - start + 1 != eeg.shape[1]: raise InferenceSchemaError("sequence range length must equal eeg.shape[1]: " + f"{end-start+1} != {eeg.shape[1]}.")
        if not np.isfinite(eeg).all(): raise InferenceSchemaError("eeg must not contain NaN or Inf.")
        return cls(SCHEMA_VERSION, sample_rate, "uV", tuple(raw_names), start, end, np.ascontiguousarray(eeg))
@dataclass(frozen=True, slots=True)
class Prediction:
    task_id: str; class_id: int; label: str; confidence: float; probabilities: tuple[float, ...]
    def to_payload(self) -> dict[str, Any]: return {"task_id": self.task_id, "class_id": self.class_id, "label": self.label, "confidence": self.confidence, "probabilities": list(self.probabilities)}
@dataclass(frozen=True, slots=True)
class EEGInferenceResponse:
    schema_version: str; sequence_start: int; sequence_end: int; predictions: tuple[Prediction, ...]; latency_ms: float
    def to_payload(self) -> dict[str, Any]: return {"schema_version": self.schema_version, "sequence_start": self.sequence_start, "sequence_end": self.sequence_end, "predictions": [p.to_payload() for p in self.predictions], "latency_ms": self.latency_ms}
def infer_eeg_window(predictor: RawWindowPredictor, *, eeg: np.ndarray, sample_rate_hz: float, channel_names: Sequence[str]) -> ThreeMentalStatePrediction:
    data = np.asarray(eeg, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"EEG must have [C, T] layout, got {data.shape}.")
    if data.shape[0] != len(channel_names):
        raise ValueError("EEG channel count does not match channel_names: " + f"{data.shape[0]} != {len(channel_names)}.")
    if data.shape[1] <= 0:
        raise ValueError("EEG must contain at least one sample.")
    if not np.isfinite(data).all():
        raise ValueError("EEG contains NaN or Inf.")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and greater than zero.")
    return predictor.predict(RawEEGWindow(data=np.ascontiguousarray(data), channel_names=[str(name) for name in channel_names], sample_rate=float(sample_rate_hz), unit="uV", layout="CT", metadata={"source": "one_window_inference"}))
def named_predictions(prediction: ThreeMentalStatePrediction) -> tuple[Prediction, ...]:
    result = []
    for field in fields(prediction):
        head = getattr(prediction, field.name)
        if not isinstance(head, HeadPrediction): raise TypeError(f"{field.name}: expected HeadPrediction, got {type(head).__name__}.")
        result.append(Prediction(field.name, int(head.label_id), str(head.label), float(head.confidence), tuple(float(v) for v in head.probabilities)))
    return tuple(result)
@dataclass(frozen=True, slots=True)
class InferenceServiceRuntime:
    predictor: RawWindowPredictor; model_package: str; device: str
def create_inference_server(host: str, port: int, runtime: InferenceServiceRuntime) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server_version = "NCCOIInference/1.0"
        def _send(self, status: HTTPStatus, body: dict[str, Any]) -> None:
            data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode(); self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self) -> None:
            self._send(HTTPStatus.OK, {"status":"ok", "model_loaded":True, "model_package":runtime.model_package, "device":runtime.device}) if self.path == "/health" else self._send(HTTPStatus.NOT_FOUND, {"error":"Not found."})
        def do_POST(self) -> None:
            if self.path != "/infer": self._send(HTTPStatus.NOT_FOUND, {"error":"Not found."}); return
            try:
                length = int(self.headers.get("Content-Length", ""));
                if length < 0: raise InferenceSchemaError("Content-Length must be non-negative.")
                if length > 64*1024*1024:
                    self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error":"Request body exceeds 64 MiB."}); return
                request = EEGInferenceRequest.from_payload(json.loads(self.rfile.read(length).decode()))
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError, InferenceSchemaError) as error: self._send(HTTPStatus.BAD_REQUEST, {"error":str(error)}); return
            try:
                started=time.perf_counter(); result=infer_eeg_window(runtime.predictor, eeg=request.eeg, sample_rate_hz=request.sample_rate_hz, channel_names=request.channel_names); latency=(time.perf_counter()-started)*1000; self._send(HTTPStatus.OK, EEGInferenceResponse(SCHEMA_VERSION, request.sequence_start, request.sequence_end, named_predictions(result), latency).to_payload())
            except ValueError as error: self._send(HTTPStatus.UNPROCESSABLE_ENTITY, {"error":str(error)})
            except Exception: logging.exception("Unhandled inference failure"); self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error":"Inference failed."})
        def log_message(self, format: str, *args: object) -> None: logging.info("%s - %s", self.address_string(), format % args)
    return HTTPServer((host, port), Handler)
