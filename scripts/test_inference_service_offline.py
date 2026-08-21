from __future__ import annotations

"""Real-data direct-vs-HTTP equivalence check for the inference service."""

import argparse
import json
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

from _bootstrap import ROOT
from bci_dayloop.data.trial_reader import open_trial_reader
from bci_dayloop.inference.http_service import InferenceServiceRuntime, create_inference_server
from bci_dayloop.inference.inference_schema import (
    EEGInferenceRequest,
    EEGInferenceResponse,
    Prediction,
    SCHEMA_VERSION,
)
from bci_dayloop.inference.window_inference import infer_eeg_window, named_predictions
from bci_dayloop.packages import load_multi_head_runtime_package


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare one real offline EEG window through direct and HTTP inference.")
    parser.add_argument("--model-package", default="model_packages/50m_three_mental_states")
    parser.add_argument("--input-h5", default="data/processed/bnci2014_001/subject_01.h5")
    parser.add_argument("--session", default=None, help="Optional source HDF5 session; defaults to all sessions.")
    parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--server-url", default=None, help="Use an already-running /infer service instead of starting one locally.")
    parser.add_argument(
        "--input-channels",
        default=None,
        help="Optional comma-separated subset of source channel names, kept in the given order.",
    )
    parser.add_argument("--export-request", default=None, help="Write the exact HTTP request fixture to this JSON path.")
    parser.add_argument("--export-reference", default=None, help="Write the direct-inference reference response to this JSON path.")
    parser.add_argument("--export-only", action="store_true", help="Export fixtures after direct inference without starting or calling HTTP.")
    return parser


def _json_compatible(value: Any) -> Any:
    """Convert numpy containers/scalars without changing JSON numeric values."""
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def request_fixture_payload(request: EEGInferenceRequest) -> dict[str, object]:
    """Return the contract-boundary request for the exact selected raw window."""
    sample_rate_hz: int | float = float(request.sample_rate_hz)
    # omniBCI_R's v1 request type uses u32. Preserve non-integral source rates
    # when present, while writing common integral device rates as JSON integers.
    if sample_rate_hz.is_integer():
        sample_rate_hz = int(sample_rate_hz)
    return _json_compatible({
        "schema_version": request.schema_version,
        "sample_rate_hz": sample_rate_hz,
        "unit": request.unit,
        "channel_names": request.channel_names,
        "sequence_start": request.sequence_start,
        "sequence_end": request.sequence_end,
        "eeg": request.eeg,
    })


def reference_fixture_payload(
    request: EEGInferenceRequest,
    direct: tuple[Prediction, ...],
    *,
    latency_ms: float,
) -> dict[str, object]:
    """Serialize formal direct predictions in the same response schema as /infer."""
    response = EEGInferenceResponse(
        schema_version=SCHEMA_VERSION,
        sequence_start=request.sequence_start,
        sequence_end=request.sequence_end,
        predictions=direct,
        latency_ms=float(latency_ms),
    )
    return _json_compatible(response.to_payload())


def write_fixture(path_value: str, payload: Mapping[str, object]) -> Path:
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_compatible(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path.resolve()


def _print_export_summary(request: EEGInferenceRequest, direct: tuple[Prediction, ...]) -> None:
    print("Input:")
    print(f"  shape={list(request.eeg.shape)}")
    print(f"  fs={request.sample_rate_hz}")
    print(f"  unit={request.unit}")
    print(f"  sequence={request.sequence_start}..{request.sequence_end}")
    print(f"  channels={list(request.channel_names)}")
    print("Tasks:")
    for prediction in direct:
        print(f"  {prediction.task_id}")


def select_input_channels(
    eeg: np.ndarray,
    channel_names: list[str],
    selected_names: str | None,
) -> tuple[np.ndarray, list[str]]:
    """Select source channels before the shared direct/HTTP request is built."""
    if selected_names is None:
        return eeg, channel_names
    requested = [name.strip() for name in selected_names.split(",") if name.strip()]
    if not requested:
        raise ValueError("--input-channels must contain at least one channel name.")
    if len(set(requested)) != len(requested):
        raise ValueError("--input-channels must not contain duplicate channel names.")
    source_indices = {name: index for index, name in enumerate(channel_names)}
    missing = [name for name in requested if name not in source_indices]
    if missing:
        raise ValueError(
            "--input-channels contains names absent from the selected H5 window: "
            f"{missing}."
        )
    return np.ascontiguousarray(eeg[[source_indices[name] for name in requested]]), requested


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        url,
        data=json.dumps(payload, allow_nan=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        if response.status != 200:
            raise RuntimeError(f"Expected HTTP 200, got {response.status}.")
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError("Inference response must be a JSON object.")
    return result


def _assert_equivalent(direct: tuple[object, ...], response: dict[str, object]) -> float:
    actual = response.get("predictions")
    if not isinstance(actual, list) or len(actual) != len(direct):
        raise AssertionError(f"Task count differs: direct={len(direct)}, HTTP={actual!r}.")
    maximum_error = 0.0
    for expected, received in zip(direct, actual, strict=True):
        expected_id = getattr(expected, "task_id")
        if not isinstance(received, dict):
            raise AssertionError("HTTP prediction must be an object.")
        for field in ("task_id", "class_id", "label"):
            if received.get(field) != getattr(expected, field):
                raise AssertionError(
                    f"{expected_id}: {field} differs: "
                    f"direct={getattr(expected, field)!r}, HTTP={received.get(field)!r}."
                )
        expected_probabilities = np.asarray(getattr(expected, "probabilities"), dtype=np.float64)
        actual_probabilities = np.asarray(received.get("probabilities"), dtype=np.float64)
        np.testing.assert_allclose(actual_probabilities, expected_probabilities, rtol=1e-5, atol=1e-6)
        confidence_error = abs(float(received["confidence"]) - float(getattr(expected, "confidence")))
        maximum_error = max(maximum_error, confidence_error, float(np.max(np.abs(actual_probabilities - expected_probabilities))))
    return maximum_error


def main() -> None:
    args = _parser().parse_args()
    if args.export_only and not (args.export_request or args.export_reference):
        _parser().error("--export-only requires --export-request and/or --export-reference.")
    predictor = load_multi_head_runtime_package(_path(args.model_package), device=args.device)
    reader = open_trial_reader(data_reader="eeg", path=_path(args.input_h5), canonical_subject_id=1)
    source = reader.load(session=args.session)
    if not 0 <= args.trial_index < len(source["data"]):
        raise IndexError(f"trial-index must be in [0, {len(source['data']) - 1}].")
    sample_rate_hz = float(reader.metadata.sample_rate)
    required_samples = int(round(predictor.window_seconds * sample_rate_hz))
    trial = np.asarray(source["data"][args.trial_index], dtype=np.float32)
    if trial.shape[1] < required_samples:
        raise ValueError(f"Selected trial has {trial.shape[1]} samples; service package requires {required_samples}.")
    eeg, channel_names = select_input_channels(
        np.ascontiguousarray(trial[:, :required_samples]),
        list(reader.metadata.channel_names),
        args.input_channels,
    )
    request = EEGInferenceRequest.from_payload({
        "schema_version": SCHEMA_VERSION,
        "sample_rate_hz": sample_rate_hz,
        "unit": "uV",
        "channel_names": channel_names,
        "sequence_start": 10_000,
        "sequence_end": 10_000 + required_samples - 1,
        "eeg": eeg.tolist(),
    })

    # Path A uses the same public core entry point as the HTTP handler.
    direct_started = time.perf_counter()
    direct = named_predictions(infer_eeg_window(
        predictor,
        eeg=request.eeg,
        sample_rate_hz=request.sample_rate_hz,
        channel_names=request.channel_names,
    ))
    direct_latency_ms = (time.perf_counter() - direct_started) * 1000.0

    request_payload = request_fixture_payload(request)
    reference_payload = reference_fixture_payload(
        request,
        direct,
        latency_ms=direct_latency_ms,
    )
    exported = False
    if args.export_request:
        print(f"Exported request fixture:\n  {write_fixture(args.export_request, request_payload)}")
        exported = True
    if args.export_reference:
        print(f"Exported direct reference:\n  {write_fixture(args.export_reference, reference_payload)}")
        exported = True
    if exported:
        _print_export_summary(request, direct)
    if args.export_only:
        return

    server = None
    thread = None
    if args.server_url:
        infer_url = args.server_url.rstrip("/") + "/infer"
    else:
        server = create_inference_server(
            "127.0.0.1", 0,
            InferenceServiceRuntime(predictor, str(_path(args.model_package)), str(predictor.device)),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        infer_url = f"http://127.0.0.1:{server.server_port}/infer"
    try:
        response = _post_json(infer_url, request_payload)
        maximum_error = _assert_equivalent(direct, response)
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)

    print(json.dumps({
        "status": "PASS",
        "input_h5": str(_path(args.input_h5)),
        "task_count": len(direct),
        "task_ids": [item.task_id for item in direct],
        "max_confidence_or_probability_error": maximum_error,
        "service_latency_ms": response["latency_ms"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
