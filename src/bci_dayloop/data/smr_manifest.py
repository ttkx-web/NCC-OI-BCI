"""Manifest-driven, read-only SMR H5 inspection and deterministic builds.

This is deliberately a small data-management layer.  It writes the same flat
EEGHDF5 core contract consumed by existing readers; training never reads a
manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from .hdf5_dataset import EEGHDF5, HDF5Metadata
from .trial_reader import open_trial_reader
from bci_dayloop.models.model_50m.config import STANDARD_64_CHANNELS
from bci_dayloop.models.model_50m.preprocessing import canonicalize_channel_name


MANIFEST_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
VALID_SOURCE_STATUSES = {"accepted", "excluded", "duplicate", "pending"}
VALID_SESSION_STATUSES = VALID_SOURCE_STATUSES
REQUIRED_DATASETS = ("data", "labels", "subject_ids", "session_ids", "trial_ids")


def _json(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def _strings(values: Iterable[Any]) -> np.ndarray:
    return np.asarray([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values], dtype=object)


def string_array(values: Iterable[str]) -> np.ndarray:
    return np.asarray(list(values), dtype=h5py.string_dtype(encoding="utf-8"))


def trial_hash(trial: np.ndarray) -> str:
    """Stable digest of canonical float32 [C,T] bytes."""
    signal = np.ascontiguousarray(np.asarray(trial, dtype=np.float32))
    return hashlib.sha256(signal.view(np.uint8)).hexdigest()


def resolve_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def portable_path(manifest_path: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(manifest_path.parent.resolve()))
    except ValueError:
        return str(path.resolve())


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported or missing SMR manifest schema_version.")
    contract = manifest.get("canonical_contract")
    if not isinstance(contract, dict):
        raise ValueError("Manifest is missing canonical_contract.")
    required = ("dataset_name", "canonical_subject_name", "canonical_subject_id", "class_names", "sample_rate", "unit", "window_seconds", "samples_per_trial", "channel_names", "allowed_auxiliary_channels")
    missing = [name for name in required if name not in contract]
    if missing:
        raise ValueError(f"Manifest canonical_contract missing {missing}.")
    if int(contract["samples_per_trial"]) != round(float(contract["sample_rate"]) * float(contract["window_seconds"])):
        raise ValueError("Manifest samples_per_trial does not equal sample_rate * window_seconds.")
    if not contract["channel_names"] or len(set(contract["channel_names"])) != len(contract["channel_names"]):
        raise ValueError("Manifest canonical channel order is empty or duplicated.")
    if not isinstance(manifest.get("source_files"), list) or not isinstance(manifest.get("sessions"), list):
        raise ValueError("Manifest must contain source_files and sessions lists.")
    seen_sessions: set[str] = set()
    for source in manifest["source_files"]:
        if source.get("status") not in VALID_SOURCE_STATUSES or not source.get("path"):
            raise ValueError("Invalid source file manifest entry.")
    for session in manifest["sessions"]:
        if session.get("status") not in VALID_SESSION_STATUSES or not session.get("source_file") or not session.get("source_session_id"):
            raise ValueError("Invalid session manifest entry.")
        cid = session.get("canonical_session_id")
        if session["status"] == "accepted" and not cid:
            raise ValueError("Accepted sessions require canonical_session_id.")
        if cid:
            if cid in seen_sessions:
                raise ValueError(f"Duplicate canonical session ID {cid}.")
            seen_sessions.add(cid)


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def canonical_indices(channel_names: list[str], contract: dict[str, Any]) -> tuple[np.ndarray | None, dict[str, Any]]:
    canonical = list(contract["channel_names"])
    allowed = set(contract["allowed_auxiliary_channels"])
    raw = list(channel_names)
    auxiliary = [name for name in raw if name in allowed]
    unknown = [name for name in raw if name not in set(canonical) and name not in allowed]
    retained = [name for name in raw if name not in allowed]
    details = {
        "raw_channels": raw, "canonical_eeg_channels": retained, "extra_channels": unknown,
        "allowed_auxiliary_channels": auxiliary,
        "missing_channels": [name for name in canonical if name not in retained],
        "reordered": retained != canonical if not unknown else False,
    }
    if unknown or retained != canonical:
        return None, details
    return np.asarray([index for index, name in enumerate(raw) if name not in allowed], dtype=np.int64), details


def mapping_summary(channel_names: list[str]) -> dict[str, Any]:
    mapped: dict[str, list[str]] = {}
    ignored: list[str] = []
    for name in channel_names:
        normalized = canonicalize_channel_name(name)
        if normalized in STANDARD_64_CHANNELS:
            mapped.setdefault(normalized, []).append(name)
        else:
            ignored.append(name)
    duplicates = {name: values for name, values in mapped.items() if len(values) > 1}
    mapped_names = list(mapped)
    return {
        "mapped_channels": mapped_names, "ignored": ignored,
        "missing_standard_64": [name for name in STANDARD_64_CHANNELS if name not in mapped],
        "duplicates": duplicates, "mapped_count": len(mapped_names),
    }


def source_summary(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read the standard flat schema once and describe failures without mutation."""
    with h5py.File(path, "r") as handle:
        missing = [name for name in REQUIRED_DATASETS if name not in handle]
        attrs_missing = [name for name in ("channel_names", "class_names", "dataset_name", "sample_rate", "unit") if name not in handle.attrs]
        if missing or attrs_missing:
            return {"missing_datasets": missing, "missing_attributes": attrs_missing}, {}
        try:
            channels = _json(handle.attrs["channel_names"])
            classes = _json(handle.attrs["class_names"])
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return {"metadata_error": str(exc)}, {}
        arrays = {
            "labels": handle["labels"][:].astype(np.int64, copy=False),
            "subject_ids": handle["subject_ids"][:].astype(np.int64, copy=False),
            "session_ids": _strings(handle["session_ids"][:]),
            "trial_ids": handle["trial_ids"][:].astype(np.int64, copy=False),
        }
        data = handle["data"][:]
        arrays["data"] = data
        summary = {
            "data_shape": list(data.shape), "data_dtype": str(data.dtype), "channel_names": channels,
            "class_names": classes, "dataset_name": str(handle.attrs["dataset_name"]),
            "sample_rate": float(handle.attrs["sample_rate"]), "unit": str(handle.attrs["unit"]),
            "lengths": {"data": len(data), **{name: len(value) for name, value in arrays.items() if name != "data"}},
        }
    return summary, arrays


def _check(name: str, status: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "status": status, "message": message, "details": details or {}}


def status_of(checks: list[dict[str, Any]]) -> str:
    states = {item["status"] for item in checks}
    return "REJECT" if "REJECT" in states else "WARNING" if "WARNING" in states else "PASS"


def parse_session_datetime(value: str) -> tuple[str | None, str | None, str | None]:
    match = re.search(r"_(\d{4})_(\d{6})$", value)
    if not match:
        return None, None, None
    mmdd, clock = match.groups()
    return f"{mmdd[:2]}-{mmdd[2:]}", f"{clock[:2]}:{clock[2:4]}:{clock[4:]}", mmdd + clock


def _stats(data: np.ndarray) -> dict[str, Any]:
    data = np.asarray(data, dtype=np.float32)
    trial_p2p = np.ptp(data, axis=(1, 2))
    trial_std = np.std(data, axis=(1, 2))
    trial_abs = np.mean(np.abs(data), axis=(1, 2))
    channel_std = np.std(data, axis=(0, 2))
    channel_trial_std = np.median(np.std(data, axis=2), axis=0)
    channel_trial_p2p = np.median(np.ptp(data, axis=2), axis=0)
    summary = lambda x: {key: float(value) for key, value in zip(("median", "p95", "max"), np.percentile(x, (50, 95, 100)), strict=True)}
    return {
        "nan_count": int(np.isnan(data).sum()), "inf_count": int(np.isinf(data).sum()),
        "file": {"min": float(np.min(data)), "max": float(np.max(data)), "mean": float(np.mean(data)), "std": float(np.std(data))},
        "trial_p2p": summary(trial_p2p), "trial_std": summary(trial_std), "trial_mean_abs": summary(trial_abs),
        "per_channel": {"overall_std": channel_std.tolist(), "median_trial_std": channel_trial_std.tolist(), "median_trial_p2p": channel_trial_p2p.tolist()},
    }


def qc_reference(data: np.ndarray, channel_names: list[str]) -> dict[str, Any]:
    stats = _stats(data)
    return {"trial_p2p": stats["trial_p2p"], "trial_std": stats["trial_std"], "per_channel": stats["per_channel"], "channel_names": channel_names, "rule": "warning when new P95/median is >3x or <1/3x accepted reference; channel overall_std uses same ratio"}


def _qc_check(stats: dict[str, Any], reference: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    details: dict[str, Any] = {"reference_available": bool(reference)}
    if not reference:
        return "PASS", details
    ratios: dict[str, float] = {}
    for metric in ("trial_p2p", "trial_std"):
        for quantile in ("median", "p95"):
            baseline = float(reference[metric][quantile])
            ratios[f"{metric}_{quantile}"] = float(stats[metric][quantile]) / baseline if baseline else float("inf")
    actual = np.asarray(stats["per_channel"]["overall_std"], dtype=float)
    baseline = np.asarray(reference["per_channel"]["overall_std"], dtype=float)
    cratio = actual / np.maximum(baseline, 1e-12)
    details.update({"ratios": ratios, "channel_std_ratio_min": float(cratio.min()), "channel_std_ratio_max": float(cratio.max()), "near_constant_channels": [int(i) for i in np.flatnonzero(actual <= np.median(actual) * 0.01)], "extreme_variance_channels": [int(i) for i in np.flatnonzero(cratio > 3)]})
    suspicious = any(value > 3 or value < 1 / 3 for value in ratios.values()) or bool(details["near_constant_channels"] or details["extreme_variance_channels"])
    return ("WARNING" if suspicious else "PASS"), details


def load_hash_index(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    relative = manifest.get("trial_hash_index", "trial_hashes.json")
    path = resolve_path(manifest_path, relative)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("hashes", raw)


def inspect_h5(*, input_path: Path, manifest_path: Path, reference_canonical: Path | None = None) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    contract = manifest["canonical_contract"]
    checks: list[dict[str, Any]] = []
    summary, arrays = source_summary(input_path)
    base = {"schema_version": REPORT_SCHEMA_VERSION, "input_file": str(input_path.resolve()), "manifest": portable_path(manifest_path, manifest_path), "checks": checks, "sessions": [], "duplicates": {"new_trials": 0, "exact_duplicates": 0, "duplicate_ratio": 0.0, "items": []}}
    if not arrays:
        checks.append(_check("schema", "REJECT", "Required schema or metadata is unreadable.", summary))
        base.update({"overall_status": "REJECT", "contract": {"schema": summary}, "recommended_action": "reject"})
        return base
    data = arrays["data"]
    shape_ok = data.ndim == 3 and all(summary["lengths"][name] == len(data) for name in ("labels", "subject_ids", "session_ids", "trial_ids")) and data.shape[1] == len(summary["channel_names"])
    dtype_ok = data.dtype == np.dtype("float32")
    checks.append(_check("schema", "PASS" if shape_ok and dtype_ok else "REJECT", "Flat EEGHDF5 schema compatible." if shape_ok and dtype_ok else "Shape/dtype/trial-level alignment mismatch.", summary))
    task_ok = summary["dataset_name"] == contract["dataset_name"] and summary["class_names"] == contract["class_names"] and np.isin(arrays["labels"], np.arange(len(contract["class_names"]))).all()
    checks.append(_check("task", "PASS" if task_ok else "REJECT", "Task and label semantics match manifest." if task_ok else "dataset_name, class_names, or labels differ from manifest.", {"actual": {"dataset_name": summary["dataset_name"], "class_names": summary["class_names"], "labels": sorted(np.unique(arrays["labels"]).tolist())}, "expected": {"dataset_name": contract["dataset_name"], "class_names": contract["class_names"]}}))
    known_source_subjects = {int(subject) for source in manifest["source_files"] for subject in source.get("source_subject_ids", [])}
    actual_source_subjects = sorted(int(subject) for subject in np.unique(arrays["subject_ids"]))
    subject_changed = bool(known_source_subjects and not set(actual_source_subjects).issubset(known_source_subjects))
    checks.append(_check("source_subject", "WARNING" if subject_changed else "PASS", "Source subject ID is known." if not subject_changed else "Source subject ID is new relative to manifest history.", {"actual": actual_source_subjects, "known": sorted(known_source_subjects)}))
    window_ok = summary["sample_rate"] == float(contract["sample_rate"]) and summary["unit"] == contract["unit"] and data.ndim == 3 and data.shape[-1] == int(contract["samples_per_trial"])
    checks.append(_check("window", "PASS" if window_ok else "REJECT", "Window/sample-rate/unit match manifest." if window_ok else "Window/sample-rate/unit incompatible.", {"actual": {"sample_rate": summary["sample_rate"], "unit": summary["unit"], "samples": data.shape[-1] if data.ndim == 3 else None}, "expected": {"sample_rate": contract["sample_rate"], "unit": contract["unit"], "samples": contract["samples_per_trial"], "seconds": contract["window_seconds"]}}))
    indices, channel_details = canonical_indices(summary["channel_names"], contract)
    channel_status = "REJECT" if indices is None and (channel_details["missing_channels"] or channel_details["reordered"]) else "WARNING" if channel_details["extra_channels"] else "PASS"
    checks.append(_check("channels", channel_status, "Canonical EEG order recovered exactly." if indices is not None else "Canonical EEG order cannot be recovered without an unreviewed change.", channel_details))
    mapping = mapping_summary(channel_details["canonical_eeg_channels"])
    expected_mapping = manifest.get("mapping_reference", {})
    mapping_status = "PASS" if not expected_mapping or {key: mapping.get(key) for key in ("mapped_channels", "ignored", "missing_standard_64")} == {key: expected_mapping.get(key) for key in ("mapped_channels", "ignored", "missing_standard_64")} else "WARNING"
    checks.append(_check("50m_mapping", mapping_status, "50M mapping matches reference." if mapping_status == "PASS" else "50M mapping differs from reference.", mapping))
    canonical_data = data[:, indices, :] if indices is not None else None
    if canonical_data is not None:
        finite = bool(np.isfinite(canonical_data).all())
        checks.append(_check("signal_finite", "PASS" if finite else "REJECT", "No NaN/Inf in canonical EEG." if finite else "NaN/Inf found in canonical EEG."))
        if finite:
            qc = _stats(canonical_data)
            reference = manifest.get("qc_reference")
            if reference_canonical and not reference:
                with h5py.File(reference_canonical, "r") as handle:
                    reference = qc_reference(handle["data"][:], list(contract["channel_names"]))
            qc_status, qc_details = _qc_check(qc, reference)
            qc["relative_reference"] = qc_details
            checks.append(_check("signal_qc", qc_status, "QC compatible with accepted reference." if qc_status == "PASS" else "QC differs materially from accepted reference; review required.", qc_details))
            base["qc"] = qc
    hash_index = load_hash_index(manifest_path, manifest)
    if canonical_data is not None:
        duplicate_items: list[dict[str, Any]] = []
        new_trials = 0
        for row, signal in enumerate(canonical_data):
            digest = trial_hash(signal)
            previous = hash_index.get(digest, [])
            if previous:
                duplicate_items.append({"source_trial_id": int(arrays["trial_ids"][row]), "source_session_id": str(arrays["session_ids"][row]), "sha256": digest, "existing": previous[0]})
            else:
                new_trials += 1
        total = len(canonical_data)
        base["duplicates"] = {"new_trials": new_trials, "exact_duplicates": len(duplicate_items), "duplicate_ratio": len(duplicate_items) / total if total else 0.0, "items": duplicate_items}
        duplicate_status = "REJECT" if total and len(duplicate_items) == total else "WARNING" if duplicate_items else "PASS"
        checks.append(_check("duplicates", duplicate_status, "No exact canonical trial duplicates." if not duplicate_items else ("Input is a 100% duplicate export." if len(duplicate_items) == total else "Input includes mixed duplicate and new trials."), {"new_trials": new_trials, "exact_duplicates": len(duplicate_items)}))
    existing_ids = [int(str(entry["canonical_session_id"])[1:]) for entry in manifest["sessions"] if entry["status"] == "accepted"]
    candidate_number = max(existing_ids, default=0) + 1
    for session_id in dict.fromkeys(arrays["session_ids"].tolist()):
        rows = np.flatnonzero(arrays["session_ids"] == session_id)
        counts = np.bincount(arrays["labels"][rows], minlength=len(contract["class_names"])).tolist()
        complete = all(value > 0 for value in counts)
        balanced = len(set(counts)) == 1
        date, clock, sortable = parse_session_datetime(str(session_id))
        historical_counts = {int(entry["trial_count"]) for entry in manifest["sessions"] if entry["status"] == "accepted"}
        unusual_trial_count = bool(historical_counts and len(rows) not in historical_counts)
        state = "PASS" if complete and balanced and sortable and not unusual_trial_count else "WARNING"
        session_qc: dict[str, Any] = {}
        if canonical_data is not None and np.isfinite(canonical_data[rows]).all():
            session_stats = _stats(canonical_data[rows])
            state_qc, relative = _qc_check(session_stats, manifest.get("qc_reference"))
            session_qc = {"trial_p2p": session_stats["trial_p2p"], "trial_std": session_stats["trial_std"], "relative_reference": relative}
            if state_qc == "WARNING": state = "WARNING"
        base["sessions"].append({"source_session_id": str(session_id), "source_date": date, "source_time": clock, "sort_key": sortable, "candidate_canonical_session_id": f"S{candidate_number}", "trial_count": int(len(rows)), "class_counts": counts, "four_class_complete": complete, "balanced": balanced, "unusual_trial_count": unusual_trial_count, "status": state, "qc": session_qc})
        candidate_number += 1
    overall = status_of(checks)
    base["source_subject_ids"] = actual_source_subjects
    base["overall_status"] = overall
    base["contract"] = {"schema": checks[0]["status"], "task": checks[1]["status"], "window": checks[3]["status"], "channels": checks[4]["status"]}
    base["recommended_action"] = "reject" if overall == "REJECT" else "review" if overall == "WARNING" else "accept"
    return base


def accepted_sessions(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    sessions = [entry for entry in manifest["sessions"] if entry["status"] == "accepted"]
    return sorted(sessions, key=lambda entry: int(str(entry["canonical_session_id"])[1:]))


def _source_entry(manifest: dict[str, Any], path: str) -> dict[str, Any] | None:
    return next((source for source in manifest["source_files"] if source["path"] == path), None)


def build_canonical(*, manifest_path: Path, manifest: dict[str, Any], output: Path) -> dict[str, Any]:
    """Fresh deterministic build from accepted raw sessions only."""
    validate_manifest(manifest)
    contract = manifest["canonical_contract"]
    sessions = accepted_sessions(manifest)
    if not sessions:
        raise ValueError("Cannot build canonical H5 without accepted sessions.")
    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    source_subjects: list[np.ndarray] = []
    source_trials: list[np.ndarray] = []
    source_sessions: list[str] = []
    canon_sessions: list[str] = []
    source_file_paths: list[str] = []
    source_ids: list[int] = []
    file_mapping: dict[str, str] = {}
    source_subject_mapping: dict[str, set[int]] = {}
    for entry in sessions:
        source_path = resolve_path(manifest_path, entry["source_file"])
        summary, arrays = source_summary(source_path)
        if not arrays:
            raise ValueError(f"Accepted source unreadable: {source_path}: {summary}")
        indices, details = canonical_indices(summary["channel_names"], contract)
        if indices is None:
            raise ValueError(f"{source_path}: channel contract failed: {details}")
        if summary["dataset_name"] != contract["dataset_name"] or summary["class_names"] != contract["class_names"] or summary["sample_rate"] != float(contract["sample_rate"]) or summary["unit"] != contract["unit"]:
            raise ValueError(f"{source_path}: canonical task/window contract changed.")
        rows = np.flatnonzero(arrays["session_ids"] == entry["source_session_id"])
        if not len(rows):
            raise ValueError(f"{source_path}: accepted session absent: {entry['source_session_id']}")
        block = arrays["data"][rows][:, indices, :]
        if block.dtype != np.dtype("float32") or block.shape[1:] != (len(contract["channel_names"]), int(contract["samples_per_trial"])) or not np.isfinite(block).all():
            raise ValueError(f"{source_path}: accepted data block violates canonical contract.")
        file_id = source_file_paths.index(entry["source_file"]) if entry["source_file"] in source_file_paths else len(source_file_paths)
        if file_id == len(source_file_paths):
            source_file_paths.append(entry["source_file"])
            file_mapping[str(file_id)] = Path(entry["source_file"]).name
        blocks.append(block)
        labels.append(arrays["labels"][rows])
        source_subjects.append(arrays["subject_ids"][rows])
        source_subject_mapping.setdefault(Path(entry["source_file"]).name, set()).update(int(x) for x in arrays["subject_ids"][rows])
        source_trials.append(arrays["trial_ids"][rows])
        source_sessions.extend([entry["source_session_id"]] * len(rows))
        canon_sessions.extend([entry["canonical_session_id"]] * len(rows))
        source_ids.extend([file_id] * len(rows))
    data = np.concatenate(blocks).astype(np.float32, copy=False)
    final_labels = np.concatenate(labels).astype(np.int64, copy=False)
    hashes: dict[str, int] = {}
    for index, signal in enumerate(data):
        digest = trial_hash(signal)
        if digest in hashes:
            raise ValueError(f"Canonical duplicate invariant failed: trials {hashes[digest]} and {index}.")
        hashes[digest] = index
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.create_dataset("data", data=data, dtype="float32", compression="gzip", shuffle=True, chunks=(1, data.shape[1], data.shape[2]))
        handle.create_dataset("labels", data=final_labels, dtype="int64")
        handle.create_dataset("subject_ids", data=np.full(len(data), int(contract["canonical_subject_id"]), dtype=np.int64), dtype="int64")
        handle.create_dataset("session_ids", data=string_array(canon_sessions))
        handle.create_dataset("trial_ids", data=np.arange(len(data), dtype=np.int64), dtype="int64")
        handle.create_dataset("source_file_ids", data=np.asarray(source_ids, dtype=np.int64), dtype="int64")
        handle.create_dataset("source_trial_ids", data=np.concatenate(source_trials).astype(np.int64), dtype="int64")
        handle.create_dataset("source_subject_ids_original", data=np.concatenate(source_subjects).astype(np.int64), dtype="int64")
        handle.create_dataset("source_session_ids", data=string_array(source_sessions))
        handle.attrs["sample_rate"] = float(contract["sample_rate"])
        handle.attrs["channel_names"] = json.dumps(contract["channel_names"], ensure_ascii=False)
        handle.attrs["class_names"] = json.dumps(contract["class_names"], ensure_ascii=False)
        handle.attrs["unit"] = contract["unit"]
        handle.attrs["dataset_name"] = contract["dataset_name"]
        handle.attrs["canonical_subject_id"] = int(contract["canonical_subject_id"])
        handle.attrs["canonical_subject_name"] = contract["canonical_subject_name"]
        handle.attrs["source_file_mapping"] = json.dumps(file_mapping, sort_keys=True)
        handle.attrs["source_subject_ids"] = json.dumps({name: sorted(values) for name, values in source_subject_mapping.items()}, sort_keys=True)
        handle.attrs["session_mapping"] = json.dumps({entry["canonical_session_id"]: entry["source_session_id"] for entry in sessions}, sort_keys=True)
        handle.attrs["manifest_schema_version"] = MANIFEST_SCHEMA_VERSION
        handle.attrs["canonical_build_version"] = 1
        handle.attrs["trial_id_policy"] = "canonical_trial_ids_0_to_N_minus_1_in_stable_accepted_session_order"
    return validate_canonical(output, manifest)


def validate_canonical(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    contract = manifest["canonical_contract"]
    reader = EEGHDF5(path)
    expected_metadata = HDF5Metadata(float(contract["sample_rate"]), list(contract["channel_names"]), list(contract["class_names"]), contract["unit"], contract["dataset_name"])
    if reader.metadata != expected_metadata:
        raise ValueError("Canonical EEGHDF5 metadata validation failed.")
    with h5py.File(path, "r") as handle:
        n = len(handle["data"])
        required = list(REQUIRED_DATASETS) + ["source_file_ids", "source_trial_ids", "source_subject_ids_original", "source_session_ids"]
        if any(name not in handle for name in required):
            raise ValueError("Canonical provenance datasets missing.")
        if handle["data"].dtype != np.dtype("float32") or handle["data"].shape[1:] != (len(contract["channel_names"]), int(contract["samples_per_trial"])):
            raise ValueError("Canonical data shape/dtype invalid.")
        for name in required[1:]:
            if len(handle[name]) != n:
                raise ValueError(f"Canonical {name} length mismatch.")
        if not np.isfinite(handle["data"][:]).all() or not np.isin(handle["labels"][:], np.arange(len(contract["class_names"]))).all():
            raise ValueError("Canonical finite/label validation failed.")
        if not np.array_equal(handle["subject_ids"][:], np.full(n, int(contract["canonical_subject_id"]))) or not np.array_equal(handle["trial_ids"][:], np.arange(n)):
            raise ValueError("Canonical subject/trial ID policy failed.")
        hashes: set[str] = set()
        for trial in handle["data"]:
            digest = trial_hash(trial)
            if digest in hashes:
                raise ValueError("Canonical duplicate invariant failed.")
            hashes.add(digest)
        class_counts = np.bincount(handle["labels"][:], minlength=len(contract["class_names"])).tolist()
    trial_reader = open_trial_reader(data_reader="eeg", path=path, canonical_subject_id=int(contract["canonical_subject_id"]))
    if not np.array_equal(trial_reader.trial_metadata()["trial_ids"], np.arange(n)):
        raise ValueError("TrialReader validation failed.")
    return {"path": str(path), "trials": n, "sessions": reader.available_sessions(), "class_counts": class_counts, "reader_smoke": "PASS", "duplicate_hashes": 0}


def rebuild_hash_index(*, manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    contract = manifest["canonical_contract"]
    records: dict[str, list[dict[str, Any]]] = {}
    historical_sessions = [entry for entry in manifest["sessions"] if entry["status"] in {"accepted", "excluded"}]
    for entry in historical_sessions:
        summary, arrays = source_summary(resolve_path(manifest_path, entry["source_file"]))
        if not arrays:
            continue
        indices, _ = canonical_indices(summary["channel_names"], contract)
        if indices is None:
            continue
        for row in np.flatnonzero(arrays["session_ids"] == entry["source_session_id"]):
            digest = trial_hash(arrays["data"][row, indices, :])
            records.setdefault(digest, []).append({"canonical_trial_id": entry.get("canonical_trial_id_by_source", {}).get(str(int(arrays["trial_ids"][row]))), "source_file": entry["source_file"], "source_session_id": entry["source_session_id"], "source_trial_id": int(arrays["trial_ids"][row]), "status": entry["status"]})
    return {"schema_version": 1, "hash_algorithm": "sha256_float32_canonical_CT_bytes", "hashes": records}
