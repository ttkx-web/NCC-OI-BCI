from __future__ import annotations
"""Unified direct/package/decoder/HTTP equivalence verifier for multi-head packages."""
import argparse, json, time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
import numpy as np
from _bootstrap import ROOT
from bci_dayloop.applications.three_mental_states.contract import DEFAULT_PATHS, TASKS
from bci_dayloop.applications.three_mental_states.export_config import load_three_mental_state_export_config
from bci_dayloop.applications.three_mental_states.predictor import ThreeMentalStatePredictor
from bci_dayloop.applications.three_mental_states.service import EEGInferenceRequest, EEGInferenceResponse, Prediction, SCHEMA_VERSION, infer_eeg_window, named_predictions
from bci_dayloop.data.trial_reader import open_trial_reader
from bci_dayloop.inference import MultiHeadDecodeResult, SlidingWindowDecoder
from bci_dayloop.packages import load_inference_package
from bci_dayloop.packages.inference import THREE_MENTAL_STATES_PREDICTION_MODE
from bci_dayloop.runtime.types import RawEEGWindow
def _path(value: str) -> Path:
    path = Path(value).expanduser(); return path if path.is_absolute() else (ROOT / path).resolve()
def build_parser() -> argparse.ArgumentParser:
    export_defaults = load_three_mental_state_export_config().sources
    parser = argparse.ArgumentParser(description="Verify equivalent shared-50M multi-head inference paths.")
    parser.add_argument("--mode", choices=("direct", "package", "decoder", "http", "all"), default="all")
    parser.add_argument("--input-h5", default=DEFAULT_PATHS["input_h5"]); parser.add_argument("--session", default=DEFAULT_PATHS["session"]); parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument("--backbone-checkpoint", default=str(export_defaults.backbone_checkpoint)); parser.add_argument("--workload-head", default=str(export_defaults.workload_head)); parser.add_argument("--attention-head", default=str(export_defaults.attention_head)); parser.add_argument("--emotion-head", default=str(export_defaults.emotion_head))
    parser.add_argument("--awakening-head")
    parser.add_argument("--model-package", "--package", dest="model_package", default=DEFAULT_PATHS["model_package"]); parser.add_argument("--device", "--direct-device", dest="device", choices=("cpu", "cuda", "mps"), default="cpu"); parser.add_argument("--server-url")
    parser.add_argument("--request", help="Use an existing real request fixture instead of reading --input-h5.")
    parser.add_argument("--reference-response", help="Compare HTTP against a previously exported direct response without loading the model.")
    parser.add_argument("--input-channels", help="Optional comma-separated source channel subset."); parser.add_argument("--export-request"); parser.add_argument("--export-reference"); parser.add_argument("--export-only", action="store_true")
    return parser
def _json(value: Any) -> Any:
    if isinstance(value, np.ndarray): return _json(value.tolist())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Mapping): return {str(k): _json(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [_json(v) for v in value]
    return value
def request_fixture_payload(request: EEGInferenceRequest) -> dict[str, object]:
    rate: int|float = int(request.sample_rate_hz) if request.sample_rate_hz.is_integer() else request.sample_rate_hz
    return _json({"schema_version":request.schema_version,"sample_rate_hz":rate,"unit":request.unit,"channel_names":request.channel_names,"sequence_start":request.sequence_start,"sequence_end":request.sequence_end,"eeg":request.eeg})
def reference_fixture_payload(request: EEGInferenceRequest, direct: tuple[Prediction, ...], *, latency_ms: float) -> dict[str, object]:
    return _json(EEGInferenceResponse(SCHEMA_VERSION, request.sequence_start, request.sequence_end, direct, float(latency_ms)).to_payload())
def write_fixture(path_value: str | Path, payload: Mapping[str, object]) -> Path:
    path=Path(path_value).expanduser(); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(_json(payload),ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); return path.resolve()
def select_input_channels(eeg: np.ndarray, channel_names: list[str], selected_names: str | None) -> tuple[np.ndarray,list[str]]:
    if selected_names is None: return eeg,channel_names
    requested=[name.strip() for name in selected_names.split(",") if name.strip()]
    if not requested or len(set(requested)) != len(requested): raise ValueError("--input-channels must contain unique channel names.")
    indices={name:index for index,name in enumerate(channel_names)}; missing=[name for name in requested if name not in indices]
    if missing: raise ValueError(f"--input-channels contains names absent from the selected H5 window: {missing}.")
    return np.ascontiguousarray(eeg[[indices[name] for name in requested]]), requested
def _assert_equal(expected: object, actual: object) -> float:
    maximum=0.0
    expected_tasks = tuple(getattr(expected, "task_ids", TASKS))
    actual_tasks = tuple(getattr(actual, "task_ids", TASKS))
    if set(expected_tasks) != set(actual_tasks): raise AssertionError("Direct/package task IDs differ.")
    for task in expected_tasks:
        first,second=getattr(expected,task),getattr(actual,task)
        if (first.label_id,first.label)!=(second.label_id,second.label): raise AssertionError(f"{task}: label differs.")
        error=float(np.max(np.abs(np.asarray(first.probabilities)-np.asarray(second.probabilities)))); maximum=max(maximum,error)
        np.testing.assert_allclose(first.probabilities,second.probabilities,rtol=1e-5,atol=1e-6)
    return maximum
def _post(url: str,payload:dict[str,object])->dict[str,object]:
    request=Request(url,data=json.dumps(payload,allow_nan=False).encode(),headers={"Content-Type":"application/json"},method="POST")
    with urlopen(request,timeout=120) as response: return json.loads(response.read())
def _post_health(url: str)->dict[str,object]:
    with urlopen(url,timeout=30) as response:
        if response.status != 200: raise AssertionError(f"Server health returned HTTP {response.status}.")
        payload=json.loads(response.read())
    if payload.get("status") != "ok" or payload.get("model_loaded") is not True: raise AssertionError("Server health is not OK.")
    return payload
def _assert_http(direct: tuple[Prediction,...],response:dict[str,object],request:EEGInferenceRequest)->dict[str,float]:
    if response.get("schema_version") != request.schema_version: raise AssertionError("HTTP schema_version differs.")
    if response.get("sequence_start") != request.sequence_start or response.get("sequence_end") != request.sequence_end: raise AssertionError("HTTP sequence bounds differ.")
    actual=response.get("predictions");
    if not isinstance(actual,list) or len(actual)!=len(direct): raise AssertionError("HTTP task count differs.")
    actual_by_task={item.get("task_id"):item for item in actual if isinstance(item,dict)}
    if set(actual_by_task)!={item.task_id for item in direct}: raise AssertionError("HTTP task ID set differs.")
    errors:dict[str,float]={}
    for before in direct:
        after=actual_by_task[before.task_id]
        if not isinstance(after,dict) or any(after.get(key)!=getattr(before,key) for key in ("task_id","class_id","label")): raise AssertionError(f"{before.task_id}: HTTP identity differs.")
        probabilities=np.asarray(after["probabilities"],dtype=np.float64)
        if probabilities.ndim != 1 or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1): raise AssertionError(f"{before.task_id}: invalid HTTP probabilities.")
        if not np.isclose(probabilities.sum(),1.0,rtol=1e-5,atol=1e-6): raise AssertionError(f"{before.task_id}: HTTP probabilities do not sum to one.")
        if int(after["class_id"]) != int(probabilities.argmax()) or after["label"] != before.label: raise AssertionError(f"{before.task_id}: HTTP decision is inconsistent.")
        if not np.isclose(float(after["confidence"]),float(probabilities[int(after["class_id"])]),rtol=1e-5,atol=1e-6): raise AssertionError(f"{before.task_id}: HTTP confidence is inconsistent.")
        difference=float(np.max(np.abs(np.asarray(before.probabilities)-probabilities))); errors[before.task_id]=difference; np.testing.assert_allclose(probabilities,before.probabilities,rtol=1e-5,atol=1e-6); np.testing.assert_allclose(after["confidence"],before.confidence,rtol=1e-5,atol=1e-6)
    return errors

def _request_from_args(args: argparse.Namespace) -> tuple[EEGInferenceRequest, object | None, int | None]:
    if args.request:
        payload=json.loads(_path(args.request).read_text(encoding="utf-8"))
        return EEGInferenceRequest.from_payload(payload), None, None
    reader=open_trial_reader(data_reader="eeg",path=_path(args.input_h5),canonical_subject_id=1); source=reader.load(session=args.session)
    if not 0<=args.trial_index<len(source["data"]): raise IndexError("--trial-index is outside the requested session.")
    raw=np.asarray(source["data"][args.trial_index],dtype=np.float32); eeg,names=select_input_channels(raw,list(reader.metadata.channel_names),args.input_channels)
    request=EEGInferenceRequest.from_payload({"schema_version":SCHEMA_VERSION,"sample_rate_hz":float(reader.metadata.sample_rate),"unit":"uV","channel_names":names,"sequence_start":10000,"sequence_end":10000+eeg.shape[1]-1,"eeg":eeg.tolist()})
    return request, source, int(source["trial_ids"][args.trial_index])

def _predictions_from_response(payload: Mapping[str, Any]) -> tuple[Prediction, ...]:
    values=payload.get("predictions")
    if not isinstance(values,list): raise ValueError("Reference response has no predictions list.")
    return tuple(Prediction(str(item["task_id"]),int(item["class_id"]),str(item["label"]),float(item["confidence"]),tuple(float(value) for value in item["probabilities"])) for item in values)
def main() -> None:
    args=build_parser().parse_args(); request, source, trial_id=_request_from_args(args); eeg=request.eeg; names=list(request.channel_names)
    modes=("direct","package","decoder","http") if args.mode=="all" else (args.mode,); summary={"status":"PASS","mode":args.mode,"checks":{},"tasks":[]}
    direct=None; packaged=None
    if any(mode in modes for mode in ("direct","package","decoder")):
        direct=ThreeMentalStatePredictor.from_checkpoints(backbone_checkpoint=_path(args.backbone_checkpoint),workload_head=_path(args.workload_head),attention_head=_path(args.attention_head),emotion_head=_path(args.emotion_head),awakening_head=_path(args.awakening_head) if args.awakening_head else None,device=args.device)
        direct_result=direct.predict(RawEEGWindow(data=eeg,channel_names=names,sample_rate=request.sample_rate_hz,unit="uV")); summary["checks"]["direct"]={"prediction":_json({task:head.probabilities for task,head in direct_result.items()})}
    if "package" in modes or "decoder" in modes or ("http" in modes and not args.reference_response):
        loaded_package=load_inference_package(_path(args.model_package),device=args.device)
        if loaded_package.prediction_mode != THREE_MENTAL_STATES_PREDICTION_MODE: raise ValueError("verify_three_state_inference.py requires a multi-head package.")
        summary["tasks"]=[task.task_id for task in loaded_package.tasks]
        packaged=loaded_package.predictor
    if "package" in modes:
        assert direct is not None and packaged is not None; summary["checks"]["package"]={"max_probability_error":_assert_equal(direct_result,packaged.predict(RawEEGWindow(data=eeg,channel_names=names,sample_rate=request.sample_rate_hz,unit="uV")))}
    if "decoder" in modes:
        assert direct is not None; decoder=SlidingWindowDecoder(predictor=direct,channel_names=names,sample_rate=request.sample_rate_hz,input_unit="uV",window_sec=direct.window_seconds,step_sec=direct.window_seconds); decoded=decoder.push(eeg,trial_id=trial_id);
        if not isinstance(decoded,MultiHeadDecodeResult): raise RuntimeError("SlidingWindowDecoder did not produce a multi-head result.")
        summary["checks"]["decoder"]={"max_probability_error":_assert_equal(direct_result,decoded.prediction)}
    if "http" in modes:
        if args.reference_response:
            reference_payload=json.loads(_path(args.reference_response).read_text(encoding="utf-8")); reference=_predictions_from_response(reference_payload); elapsed=0.0; summary["tasks"]=[item.task_id for item in reference]
        else:
            assert packaged is not None; started=time.perf_counter(); reference=named_predictions(infer_eeg_window(packaged,eeg=request.eeg,sample_rate_hz=request.sample_rate_hz,channel_names=request.channel_names)); elapsed=(time.perf_counter()-started)*1000
            diagnostics=packaged.last_diagnostics
            summary["checks"]["direct_reference"]={
                "preprocessing_calls": diagnostics.preprocessing_calls,
                "backbone_forwards": diagnostics.backbone_forwards,
                "head_forwards": diagnostics.head_forwards,
                "shared_feature_shape": list(diagnostics.shared_feature_shape),
            }
        if args.export_request: write_fixture(args.export_request,request_fixture_payload(request))
        if args.export_reference: write_fixture(args.export_reference,reference_fixture_payload(request,reference,latency_ms=elapsed))
        if not args.export_only:
            if not args.server_url: raise ValueError("--server-url is required for HTTP consistency; this script never starts or stops a service.")
            health=_post_health(args.server_url.rstrip("/")+"/health")
            summary["checks"]["http"]={"server_health":health,"max_probability_error_by_task":_assert_http(reference,_post(args.server_url.rstrip("/")+"/infer",request_fixture_payload(request)),request)}
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, ensure_ascii=False, indent=2))
        raise
