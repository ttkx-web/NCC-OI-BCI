"""Standard, provenance-preserving export for legacy cue-plus-imagery trials."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from bci_dayloop.data.trial_extraction import (
    ACCURACY_SCOPE,
    ELIGIBLE_FOR_ACCURACY,
    ELIGIBLE_FOR_PURE_IMAGERY_ACCURACY,
    VISUAL_CUE_DURATION_SECONDS,
    VISUAL_CUE_PRESENT,
    WINDOW_SEMANTICS,
    EEGTrial,
)


_LABELS = ("left_hand", "right_hand", "feet", "tongue")


def _strings(values: Iterable[object]) -> np.ndarray:
    return np.asarray(["" if value is None else str(value) for value in values], dtype=h5py.string_dtype("utf-8"))


def export_stage2a_trials_hdf5(
    path: str | Path,
    trials: tuple[EEGTrial, ...],
    *,
    channel_names: tuple[str, ...],
    sampling_rate: float,
    subject_id: str | None,
    session_id: str | None,
) -> dict[str, object]:
    """Write exact 4 s legacy trials without resampling, padding, or preprocessing."""
    if sampling_rate != 250:
        raise ValueError("Stage 2A export requires sampling_rate == 250")
    if not trials:
        raise ValueError("Stage 2A export requires at least one trial")
    data = np.stack([trial.eeg for trial in trials]).astype(np.float32, copy=False)
    if data.ndim != 3 or data.shape[1] != len(channel_names) or data.shape[2] != 1000:
        raise ValueError("Stage 2A export requires data with shape [N,C,1000]")
    if not np.isfinite(data).all():
        raise ValueError("Stage 2A export rejects NaN or Inf")
    if any(trial.label not in _LABELS for trial in trials):
        raise ValueError("Stage 2A export contains an unsupported label")

    required_provenance = ("bdf_sha256", "csv_sha256", "source_format", "reader_name", "unit_evidence_level")
    for trial in trials:
        missing = [field for field in required_provenance if not trial.source_metadata.get(field)]
        if missing:
            raise ValueError(f"Trial provenance missing: {', '.join(missing)}")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, "w") as handle:
        handle.create_dataset("eeg", data=data, dtype="float32", compression="gzip", shuffle=True)
        handle.create_dataset("labels", data=_strings(trial.label for trial in trials))
        handle.create_dataset("block_ids", data=_strings(trial.block_id for trial in trials))
        handle.create_dataset("trial_ids", data=_strings(trial.trial_id for trial in trials))
        handle.create_dataset("canonical_start_samples", data=np.asarray([trial.canonical_start_sample for trial in trials], dtype=np.int64))
        handle.create_dataset("canonical_end_samples", data=np.asarray([trial.canonical_end_sample for trial in trials], dtype=np.int64))
        handle.create_dataset("observed_rest_samples", data=np.asarray([trial.observed_rest_sample for trial in trials], dtype=np.int64))
        handle.create_dataset("observed_event_n_samples", data=np.asarray([trial.observed_event_n_samples for trial in trials], dtype=np.int64))
        handle.create_dataset("rest_offset_samples", data=np.asarray([trial.rest_offset_samples for trial in trials], dtype=np.int64))
        handle.create_dataset("rest_offset_seconds", data=np.asarray([trial.rest_offset_seconds for trial in trials], dtype=np.float64))
        handle.create_dataset("endpoint_qc_passed", data=np.asarray([trial.endpoint_qc_passed for trial in trials], dtype=bool))
        handle.create_dataset("window_semantics", data=_strings(trial.window_semantics for trial in trials))
        handle.create_dataset("eligible_for_accuracy", data=np.asarray([trial.eligible_for_accuracy for trial in trials], dtype=bool))
        handle.create_dataset("accuracy_scope", data=_strings(trial.accuracy_scope for trial in trials))
        handle.create_dataset("visual_cue_present", data=np.asarray([trial.visual_cue_present for trial in trials], dtype=bool))
        handle.create_dataset(
            "visual_cue_duration_seconds",
            data=np.asarray([trial.visual_cue_duration_seconds for trial in trials], dtype=np.float64),
        )
        handle.create_dataset(
            "eligible_for_pure_imagery_accuracy",
            data=np.asarray(
                [trial.eligible_for_pure_imagery_accuracy for trial in trials], dtype=bool
            ),
        )
        handle.create_dataset("extraction_policy", data=_strings(trial.extraction_policy for trial in trials))
        handle.create_dataset("start_sample", data=np.asarray([trial.start_sample for trial in trials], dtype=np.int64))
        handle.create_dataset("end_sample", data=np.asarray([trial.end_sample for trial in trials], dtype=np.int64))
        handle.create_dataset("duration_seconds", data=np.asarray([trial.duration_seconds for trial in trials], dtype=np.float64))
        for field in (
            "bdf_sha256",
            "csv_sha256",
            "source_format",
            "conversion_tool",
            "conversion_tool_version",
            "reader_name",
            "reader_version",
            "unit_evidence_level",
        ):
            handle.create_dataset(field, data=_strings(trial.source_metadata.get(field) for trial in trials))
        handle.attrs["sampling_rate"] = 250.0
        handle.attrs["unit"] = "uV"
        handle.attrs["channel_names"] = json.dumps(list(channel_names), ensure_ascii=False)
        handle.attrs["subject_id"] = subject_id or ""
        handle.attrs["session_id"] = session_id or ""
        handle.attrs["window_semantics"] = WINDOW_SEMANTICS
        handle.attrs["eligible_for_accuracy"] = ELIGIBLE_FOR_ACCURACY
        handle.attrs["accuracy_scope"] = ACCURACY_SCOPE
        handle.attrs["visual_cue_present"] = VISUAL_CUE_PRESENT
        handle.attrs["visual_cue_duration_seconds"] = VISUAL_CUE_DURATION_SECONDS
        handle.attrs["eligible_for_pure_imagery_accuracy"] = ELIGIBLE_FOR_PURE_IMAGERY_ACCURACY
    return {
        "filename": target.name,
        "trial_count": len(trials),
        "channel_count": len(channel_names),
        "sample_count": 1000,
        "sampling_rate": 250.0,
        "unit": "uV",
        "window_semantics": WINDOW_SEMANTICS,
        "eligible_for_accuracy": ELIGIBLE_FOR_ACCURACY,
        "accuracy_scope": ACCURACY_SCOPE,
        "visual_cue_present": VISUAL_CUE_PRESENT,
        "visual_cue_duration_seconds": VISUAL_CUE_DURATION_SECONDS,
        "eligible_for_pure_imagery_accuracy": ELIGIBLE_FOR_PURE_IMAGERY_ACCURACY,
        "bdf_sha256": trials[0].source_metadata["bdf_sha256"],
        "csv_sha256": trials[0].source_metadata["csv_sha256"],
    }


def read_stage2a_trials_hdf5(path: str | Path) -> dict[str, object]:
    """Read every Stage 2A export field for explicit round-trip verification."""
    with h5py.File(Path(path), "r") as handle:
        values: dict[str, object] = {
            "eeg": handle["eeg"][:].astype(np.float32, copy=False),
            "labels": handle["labels"].asstr()[:],
            "block_ids": handle["block_ids"].asstr()[:],
            "trial_ids": handle["trial_ids"].asstr()[:],
            "canonical_start_samples": handle["canonical_start_samples"][:],
            "canonical_end_samples": handle["canonical_end_samples"][:],
            "observed_rest_samples": handle["observed_rest_samples"][:],
            "observed_event_n_samples": handle["observed_event_n_samples"][:],
            "rest_offset_samples": handle["rest_offset_samples"][:],
            "rest_offset_seconds": handle["rest_offset_seconds"][:],
            "endpoint_qc_passed": handle["endpoint_qc_passed"][:],
            "window_semantics_per_trial": handle["window_semantics"].asstr()[:],
            "eligible_for_accuracy_per_trial": handle["eligible_for_accuracy"][:],
            "accuracy_scope_per_trial": handle["accuracy_scope"].asstr()[:],
            "visual_cue_present_per_trial": handle["visual_cue_present"][:],
            "visual_cue_duration_seconds_per_trial": handle[
                "visual_cue_duration_seconds"
            ][:],
            "eligible_for_pure_imagery_accuracy_per_trial": handle[
                "eligible_for_pure_imagery_accuracy"
            ][:],
            "extraction_policy": handle["extraction_policy"].asstr()[:],
            "start_sample": handle["start_sample"][:],
            "end_sample": handle["end_sample"][:],
            "duration_seconds": handle["duration_seconds"][:],
            "channel_names": json.loads(handle.attrs["channel_names"]),
            "subject_id": str(handle.attrs["subject_id"]),
            "session_id": str(handle.attrs["session_id"]),
            "sampling_rate": float(handle.attrs["sampling_rate"]),
            "unit": str(handle.attrs["unit"]),
            "window_semantics": str(handle.attrs["window_semantics"]),
            "eligible_for_accuracy": bool(handle.attrs["eligible_for_accuracy"]),
            "accuracy_scope": str(handle.attrs["accuracy_scope"]),
            "visual_cue_present": bool(handle.attrs["visual_cue_present"]),
            "visual_cue_duration_seconds": float(handle.attrs["visual_cue_duration_seconds"]),
            "eligible_for_pure_imagery_accuracy": bool(
                handle.attrs["eligible_for_pure_imagery_accuracy"]
            ),
        }
        for field in (
            "bdf_sha256",
            "csv_sha256",
            "source_format",
            "conversion_tool",
            "conversion_tool_version",
            "reader_name",
            "reader_version",
            "unit_evidence_level",
        ):
            values[field] = handle[field].asstr()[:]
        return values
