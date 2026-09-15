from __future__ import annotations

"""Convert valid ten-second Awakening Lance rows into five in-row 2 s HDF5 windows."""

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:  # Script execution.
    from _bootstrap import ROOT
except ModuleNotFoundError:  # Imported by tests.
    from scripts._bootstrap import ROOT

from bci_dayloop.data.awakening import (
    CHANNEL_NAMES,
    CLASS_NAMES,
    LABEL_MAPPING,
    MIN_ALL_CHANNEL_ZERO_RUN_SAMPLES,
    N_CHANNELS,
    PARENT_ELEMENTS,
    PARENT_SAMPLES,
    SAMPLE_RATE,
    SESSION_SEMANTICS,
    STEP_SECONDS,
    SUBWINDOWS_PER_PARENT,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
    AwakeningHDF5Writer,
    ParentWindowPlan,
    assign_logical_sessions,
    canonical_subject_mapping,
    counts_by_subject_label_session,
    extract_subwindows,
    inspect_lance_row,
    validate_awakening_hdf5,
    validate_fixed_window_contract,
)


REQUIRED_COLUMNS = (
    "sample_id", "subject_id", "data", "shape", "original_shape",
    "valid_length", "qc_pass", "label", "global_idx", "channel_names",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split each valid Awakening [62,2000] Lance parent row into five "
            "non-overlapping [62,400] HDF5 windows. Parent rows are never joined."
        )
    )
    parser.add_argument("--lance-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--step-seconds", type=float, default=STEP_SECONDS)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _import_lance() -> Any:
    try:
        import lance  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Reading Awakening requires the 'pylance' package. Install the "
            "data-preparation dependencies from requirements-data.txt."
        ) from error
    return lance


def _list_scalar_to_numpy(column: Any, index: int) -> np.ndarray:
    scalar = column[index]
    if not scalar.is_valid:
        return np.empty((0,), dtype=np.float32)
    return scalar.values.to_numpy(zero_copy_only=False)


def _scalar(column: Any, index: int) -> Any:
    return column[index].as_py()


def scan_lance(
    dataset: Any,
    *,
    batch_size: int,
) -> tuple[list[ParentWindowPlan], dict[str, Any]]:
    """Inspect source rows in bounded batches and retain only a lightweight plan."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    schema_names = set(dataset.schema.names)
    missing = set(REQUIRED_COLUMNS) - schema_names
    if missing:
        raise ValueError(f"Lance schema is missing required columns: {sorted(missing)}.")

    parents: list[ParentWindowPlan] = []
    rejection_counts: Counter[str] = Counter()
    input_label_counts: Counter[int] = Counter()
    accepted_parent_label_counts: Counter[int] = Counter()
    subject_input_counts: Counter[str] = Counter()
    zero_interval_windows = 0
    zero_interval_rows = 0
    fully_normal_rows = 0
    qc_pass_counts: Counter[str] = Counter()
    seen_global_indices: set[int] = set()
    seen_sample_ids: set[str] = set()
    row_position = 0

    scanner = dataset.scanner(columns=list(REQUIRED_COLUMNS), batch_size=batch_size)
    for batch_number, batch in enumerate(scanner.to_batches(), start=1):
        data_column = batch["data"]
        for index in range(batch.num_rows):
            current_position = row_position
            row_position += 1
            sample_id = str(_scalar(batch["sample_id"], index) or "").strip()
            source_subject_id = str(_scalar(batch["subject_id"], index) or "").strip()
            try:
                global_idx = int(_scalar(batch["global_idx"], index))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"row position {current_position}: invalid global_idx."
                ) from error
            if global_idx in seen_global_indices:
                raise ValueError(f"duplicate global_idx in Lance: {global_idx}.")
            if not sample_id:
                rejection_counts["missing_sample_id"] += 1
                continue
            if sample_id in seen_sample_ids:
                raise ValueError(f"duplicate sample_id in Lance: {sample_id!r}.")
            seen_global_indices.add(global_idx)
            seen_sample_ids.add(sample_id)
            if not source_subject_id:
                rejection_counts["missing_subject_id"] += 1
                continue

            label_value = _scalar(batch["label"], index)
            try:
                label = int(label_value)
                input_label_counts[label] += 1
            except (TypeError, ValueError):
                label = -1
            subject_input_counts[source_subject_id] += 1
            qc_pass = bool(_scalar(batch["qc_pass"], index))
            qc_pass_counts[str(qc_pass).lower()] += 1
            channel_names = _scalar(batch["channel_names"], index)
            inspection = inspect_lance_row(
                data=_list_scalar_to_numpy(data_column, index),
                shape=_scalar(batch["shape"], index),
                original_shape=_scalar(batch["original_shape"], index),
                valid_length=_scalar(batch["valid_length"], index),
                channel_names=channel_names,
                label=label_value,
            )
            if inspection.row_rejection_reason is not None:
                rejection_counts[inspection.row_rejection_reason] += 1
                continue
            if inspection.zero_run_subwindow_indices:
                zero_interval_rows += 1
                zero_interval_windows += len(inspection.zero_run_subwindow_indices)
            else:
                fully_normal_rows += 1
            if not inspection.valid_subwindow_indices:
                rejection_counts["no_valid_subwindows"] += 1
                continue
            accepted_parent_label_counts[label] += 1
            parents.append(
                ParentWindowPlan(
                    row_position=current_position,
                    global_idx=global_idx,
                    sample_id=sample_id,
                    source_subject_id=source_subject_id,
                    label=label,
                    valid_subwindow_indices=inspection.valid_subwindow_indices,
                    zero_run_subwindow_indices=inspection.zero_run_subwindow_indices,
                )
            )
        if batch_number == 1 or batch_number % 10 == 0:
            print(
                f"[scan] batches={batch_number} rows={row_position} "
                f"eligible_parents={len(parents)}",
                flush=True,
            )

    output_windows = sum(len(parent.valid_subwindow_indices) for parent in parents)
    structural_rows = fully_normal_rows + zero_interval_rows
    report: dict[str, Any] = {
        "input_rows": row_position,
        "input_label_counts": {
            str(key): int(value) for key, value in sorted(input_label_counts.items())
        },
        "input_subject_count": len(subject_input_counts),
        "input_rows_by_subject": dict(sorted(subject_input_counts.items())),
        "qc_pass_counts": dict(sorted(qc_pass_counts.items())),
        "structurally_valid_rows": structural_rows,
        "fully_normal_rows": fully_normal_rows,
        "zero_interval_affected_rows": zero_interval_rows,
        "eligible_parent_rows": len(parents),
        "accepted_parent_label_counts": {
            str(key): int(value)
            for key, value in sorted(accepted_parent_label_counts.items())
        },
        "rejected_parent_rows_by_reason": dict(sorted(rejection_counts.items())),
        "candidate_windows_from_structurally_valid_rows": (
            structural_rows * SUBWINDOWS_PER_PARENT
        ),
        "discarded_subwindows_all_channel_zero_run": zero_interval_windows,
        "output_windows": output_windows,
        "all_channel_zero_run_min_samples": MIN_ALL_CHANNEL_ZERO_RUN_SAMPLES,
    }
    return parents, report


def build_plan(
    dataset: Any,
    *,
    batch_size: int,
    validation_fraction: float,
    seed: int,
) -> tuple[
    list[ParentWindowPlan],
    dict[int, str],
    dict[str, int],
    dict[str, Any],
    dict[str, Any],
]:
    parents, report = scan_lance(dataset, batch_size=batch_size)
    if not parents:
        raise ValueError("No usable Awakening parent rows remain after validation.")
    assignments, manifest_groups = assign_logical_sessions(
        parents, validation_fraction=validation_fraction, seed=seed
    )
    mapping = canonical_subject_mapping(parents)
    counts = counts_by_subject_label_session(parents, assignments)
    for subject_id, label_counts in counts.items():
        for label, session_counts in label_counts.items():
            if not session_counts["S1"] or not session_counts["S2"]:
                raise ValueError(
                    f"split cannot preserve both sessions for subject={subject_id} "
                    f"label={label}: {session_counts}."
                )
    split_manifest = {
        "schema_version": 1,
        "session_semantics": SESSION_SEMANTICS,
        "original_experiment_session": False,
        "split_unit": "source_lance_row_id",
        "stratification": "within_each_source_subject_id_and_label",
        "seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "groups": manifest_groups,
    }
    report.update(
        {
            "sample_rate": SAMPLE_RATE,
            "window_seconds": WINDOW_SECONDS,
            "step_seconds": STEP_SECONDS,
            "samples_per_window": WINDOW_SAMPLES,
            "channels": N_CHANNELS,
            "class_names": list(CLASS_NAMES),
            "label_mapping": LABEL_MAPPING,
            "session_semantics": SESSION_SEMANTICS,
            "split_seed": int(seed),
            "validation_fraction": float(validation_fraction),
            "counts_by_subject_label_session": counts,
            "subject_mapping": mapping,
        }
    )
    return parents, assignments, mapping, split_manifest, report


def _nearest_existing_path(path: Path) -> Path:
    current = path.resolve()
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(f"No existing parent for output path: {path}.")
        current = parent
    return current


def check_disk_space(output_dir: Path, *, output_windows: int) -> dict[str, int]:
    raw_data_bytes = output_windows * N_CHANNELS * WINDOW_SAMPLES * 4
    estimated_required_bytes = int(raw_data_bytes * 1.20) + 32 * 1024 * 1024
    usage = shutil.disk_usage(_nearest_existing_path(output_dir))
    if usage.free < estimated_required_bytes:
        raise OSError(
            "Insufficient free space for Awakening conversion: "
            f"need at least {estimated_required_bytes} bytes, have {usage.free}."
        )
    return {
        "raw_data_bytes": raw_data_bytes,
        "estimated_required_bytes": estimated_required_bytes,
        "free_bytes_before_write": usage.free,
    }


def _json_temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _subject_output_paths(
    output_dir: Path, mapping: dict[str, int]
) -> dict[str, Path]:
    return {
        source_id: output_dir / f"subject_{canonical_id:02d}.h5"
        for source_id, canonical_id in mapping.items()
    }


def _preflight_targets(targets: Iterable[Path], *, overwrite: bool) -> None:
    existing = sorted(path for path in targets if path.exists())
    if existing and not overwrite:
        preview = ", ".join(str(path) for path in existing[:5])
        raise FileExistsError(
            f"Output already exists ({preview}). Pass --overwrite to replace it."
        )


def _preflight_existing_output_dir(
    output_dir: Path, *, overwrite: bool, dry_run: bool
) -> None:
    """Fail before scanning when a formal run would overwrite known outputs."""

    if overwrite or dry_run or not output_dir.exists():
        return
    known = sorted(output_dir.glob("subject_*.h5"))
    known.extend(
        path
        for path in (
            output_dir / "subject_mapping.json",
            output_dir / "split_manifest.json",
            output_dir / "conversion_report.json",
        )
        if path.exists()
    )
    if known:
        preview = ", ".join(str(path) for path in known[:5])
        raise FileExistsError(
            f"Output already exists ({preview}). Pass --overwrite to replace it."
        )


def write_conversion(
    dataset: Any,
    *,
    output_dir: Path,
    parents: Sequence[ParentWindowPlan],
    assignments: dict[int, str],
    mapping: dict[str, int],
    split_manifest: dict[str, Any],
    report: dict[str, Any],
    batch_size: int,
    validation_fraction: float,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    output_paths = _subject_output_paths(output_dir, mapping)
    json_paths = {
        "subject_mapping": output_dir / "subject_mapping.json",
        "split_manifest": output_dir / "split_manifest.json",
        "conversion_report": output_dir / "conversion_report.json",
    }
    _preflight_targets(
        [*output_paths.values(), *json_paths.values()], overwrite=overwrite
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    disk = check_disk_space(output_dir, output_windows=int(report["output_windows"]))
    report["disk_space"] = disk
    print(
        f"[disk] free={disk['free_bytes_before_write'] / 1024**3:.2f} GiB "
        f"required_estimate={disk['estimated_required_bytes'] / 1024**3:.2f} GiB",
        flush=True,
    )

    windows_by_subject = Counter()
    for parent in parents:
        windows_by_subject[parent.source_subject_id] += len(
            parent.valid_subwindow_indices
        )

    temp_h5_paths = {
        source_id: target.with_name(f".{target.name}.{os.getpid()}.tmp")
        for source_id, target in output_paths.items()
    }
    temp_json_paths = {
        name: _json_temporary(path) for name, path in json_paths.items()
    }
    for temporary in [*temp_h5_paths.values(), *temp_json_paths.values()]:
        temporary.unlink(missing_ok=True)

    writers: dict[str, AwakeningHDF5Writer] = {}
    try:
        for source_id, canonical_id in mapping.items():
            writers[source_id] = AwakeningHDF5Writer(
                temp_h5_paths[source_id],
                total_windows=int(windows_by_subject[source_id]),
                canonical_subject_id=canonical_id,
                source_subject_id=source_id,
                seed=seed,
                validation_fraction=validation_fraction,
            )

        plan_by_global_idx = {parent.global_idx: parent for parent in parents}
        ordered = sorted(parents, key=lambda parent: parent.global_idx)
        for start in range(0, len(ordered), batch_size):
            selected = ordered[start:start + batch_size]
            positions = [parent.row_position for parent in selected]
            table = dataset.take(positions, columns=list(REQUIRED_COLUMNS))
            for batch in table.combine_chunks().to_batches():
                data_column = batch["data"]
                for index in range(batch.num_rows):
                    global_idx = int(_scalar(batch["global_idx"], index))
                    parent = plan_by_global_idx[global_idx]
                    data = _list_scalar_to_numpy(data_column, index)
                    inspection = inspect_lance_row(
                        data=data,
                        shape=_scalar(batch["shape"], index),
                        original_shape=_scalar(batch["original_shape"], index),
                        valid_length=_scalar(batch["valid_length"], index),
                        channel_names=_scalar(batch["channel_names"], index),
                        label=_scalar(batch["label"], index),
                    )
                    if (
                        inspection.row_rejection_reason is not None
                        or inspection.valid_subwindow_indices
                        != parent.valid_subwindow_indices
                    ):
                        raise RuntimeError(
                            f"source row {global_idx} changed between scan and write."
                        )
                    windows = extract_subwindows(
                        data, parent.valid_subwindow_indices
                    )
                    writers[parent.source_subject_id].append(
                        windows=windows,
                        label=parent.label,
                        session_id=assignments[parent.global_idx],
                        source_lance_row_id=parent.global_idx,
                        source_subwindow_indices=parent.valid_subwindow_indices,
                        source_sample_id=parent.sample_id,
                    )
            if start == 0 or start + batch_size >= len(ordered) or (
                start // batch_size + 1
            ) % 10 == 0:
                print(
                    f"[write] parents={min(start + batch_size, len(ordered))}/"
                    f"{len(ordered)}",
                    flush=True,
                )

        for writer in writers.values():
            writer.close()

        validation: dict[str, Any] = {}
        for source_id, canonical_id in mapping.items():
            validation[source_id] = validate_awakening_hdf5(
                temp_h5_paths[source_id],
                expected_windows=int(windows_by_subject[source_id]),
                canonical_subject_id=canonical_id,
                source_subject_id=source_id,
                io_batch_size=batch_size,
            )
        report["h5_validation"] = validation
        report["output_files"] = {
            source_id: str(path)
            for source_id, path in sorted(output_paths.items())
        }
        report["output_h5_schema"] = {
            "data": "float32 [N,62,400]",
            "labels": "int64 [N]",
            "subject_ids": "int64 [N]",
            "session_ids": "UTF-8 [N]",
            "trial_ids": "int64 [N]",
            "source_lance_row_ids": "int64 [N]",
            "source_subwindow_indices": "int64 [N]",
            "source_sample_ids": "UTF-8 [N]",
        }
        subject_mapping_payload = {
            "schema_version": 1,
            "source_to_canonical": mapping,
            "canonical_to_source": {
                str(canonical): source for source, canonical in mapping.items()
            },
        }
        _write_json(temp_json_paths["subject_mapping"], subject_mapping_payload)
        _write_json(temp_json_paths["split_manifest"], split_manifest)
        _write_json(temp_json_paths["conversion_report"], report)

        for source_id, target in output_paths.items():
            os.replace(temp_h5_paths[source_id], target)
        for name, target in json_paths.items():
            os.replace(temp_json_paths[name], target)
    except Exception:
        for writer in writers.values():
            writer.abort()
        for temporary in [*temp_h5_paths.values(), *temp_json_paths.values()]:
            temporary.unlink(missing_ok=True)
        raise

    report["output_bytes"] = sum(
        path.stat().st_size for path in output_paths.values()
    )
    # Refresh the report atomically with final on-disk byte counts.
    report_temp = _json_temporary(json_paths["conversion_report"])
    _write_json(report_temp, report)
    os.replace(report_temp, json_paths["conversion_report"])
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_fixed_window_contract(
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
    )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be strictly between 0 and 1.")
    lance_path = args.lance_path.resolve()
    output_dir = args.output_dir.resolve()
    if not lance_path.exists():
        raise FileNotFoundError(f"Awakening Lance dataset not found: {lance_path}")
    _preflight_existing_output_dir(
        output_dir, overwrite=args.overwrite, dry_run=args.dry_run
    )

    lance = _import_lance()
    dataset = lance.dataset(str(lance_path))
    parents, assignments, mapping, split_manifest, report = build_plan(
        dataset,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    report["lance_path"] = str(lance_path)
    report["lance_version"] = int(dataset.version)
    report["dry_run"] = bool(args.dry_run)
    if args.dry_run:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    report["dry_run"] = False
    final_report = write_conversion(
        dataset,
        output_dir=output_dir,
        parents=parents,
        assignments=assignments,
        mapping=mapping,
        split_manifest=split_manifest,
        report=report,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(final_report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
