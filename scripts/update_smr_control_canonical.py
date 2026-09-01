from __future__ import annotations

"""Review an inspection report, update its manifest, and rebuild canonical H5."""

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path

import h5py

try:
    from _bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:
    from scripts._bootstrap import ROOT  # noqa: F401

from bci_dayloop.data.smr_manifest import (
    accepted_sessions,
    build_canonical,
    load_manifest,
    portable_path,
    qc_reference,
    rebuild_hash_index,
    resolve_path,
    validate_manifest,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply reviewed SMR inspection data through a deterministic canonical rebuild.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inspection-report", type=Path, required=True)
    parser.add_argument("--accept-session", action="append", default=[], metavar="SOURCE_SESSION_ID")
    parser.add_argument("--accept-all-pass", action="store_true", help="Accept only sessions whose inspection status is PASS.")
    parser.add_argument("--review-note", default=None, help="Persisted for explicitly accepted WARNING sessions.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser.parse_args()


def _canonical_output(manifest_path: Path, manifest: dict) -> Path:
    return resolve_path(manifest_path, manifest["canonical_output"])


def _next_session_id(manifest: dict) -> int:
    current = [int(str(item["canonical_session_id"])[1:]) for item in accepted_sessions(manifest)]
    return max(current, default=0) + 1


def propose(manifest_path: Path, manifest: dict, report_path: Path, report: dict, args: argparse.Namespace) -> tuple[dict, list[dict], list[dict]]:
    if report.get("schema_version") != 1:
        raise ValueError("Unsupported inspection report schema.")
    source_path = Path(report["input_file"]).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Inspection source no longer exists: {source_path}")
    if report.get("overall_status") == "REJECT":
        raise ValueError("REJECT inspection reports cannot be ingested.")
    proposed = copy.deepcopy(manifest)
    source_rel = portable_path(manifest_path, source_path)
    existing = next((item for item in proposed["source_files"] if item["path"] == source_rel), None)
    if existing and existing.get("status") in {"accepted", "duplicate"}:
        raise ValueError(f"Source is already recorded as {existing['status']}: {source_rel}")
    explicit = set(args.accept_session)
    known = {session["source_session_id"] for session in report["sessions"]}
    unknown = explicit - known
    if unknown:
        raise ValueError(f"--accept-session not present in report: {sorted(unknown)}")
    accept = {session["source_session_id"] for session in report["sessions"] if args.accept_all_pass and session["status"] == "PASS"}
    accept.update(explicit)
    if not accept:
        raise ValueError("No sessions selected. Use --accept-all-pass or --accept-session.")
    next_id = _next_session_id(proposed)
    existing_sort_keys = [entry.get("source_sort_key") for entry in accepted_sessions(proposed) if entry.get("source_sort_key")]
    latest_sort_key = max(existing_sort_keys) if existing_sort_keys else None
    accepted: list[dict] = []
    pending: list[dict] = []
    already = {(entry["source_file"], entry["source_session_id"]) for entry in proposed["sessions"]}
    # New accepted sessions are assigned chronologically among themselves.  Old
    # S IDs are never renumbered; an earlier source requires explicit review.
    for session in sorted(report["sessions"], key=lambda item: item.get("sort_key") or "9999999999"):
        key = (source_rel, session["source_session_id"])
        if key in already:
            raise ValueError(f"Session is already in manifest: {key}")
        entry = {
            "source_file": source_rel, "source_session_id": session["source_session_id"], "source_date": session.get("source_date"), "source_time": session.get("source_time"), "source_sort_key": session.get("sort_key"), "trial_count": session["trial_count"], "class_counts": session["class_counts"],
        }
        early = bool(session.get("sort_key") and latest_sort_key and session["sort_key"] < latest_sort_key)
        selected = session["source_session_id"] in accept
        # A chronological insertion cannot silently renumber stable S IDs.
        # Explicit selection is the required human decision for this case.
        if early and session["source_session_id"] not in explicit:
            selected = False
        if selected:
            entry.update({"status": "accepted", "canonical_session_id": f"S{next_id}", "qc_status": "pass" if session["status"] == "PASS" else "warning_accepted", "inspection_report": portable_path(manifest_path, report_path)})
            if early:
                entry["ordering_warning"] = "Source timestamp predates an existing canonical session; retained historical S IDs and appended after explicit review."
            if session["status"] != "PASS" and args.review_note:
                entry["review_note"] = args.review_note
            accepted.append(entry)
            next_id += 1
        else:
            entry.update({"status": "pending", "inspection_report": portable_path(manifest_path, report_path), "reason": "earlier_than_existing_canonical_session_requires_explicit_accept" if early else "not_selected_after_inspection"})
            pending.append(entry)
        proposed["sessions"].append(entry)
    source_status = "accepted" if accepted else "pending"
    source_entry = {"path": source_rel, "status": source_status, "inspection_report": portable_path(manifest_path, report_path), "source_subject_ids": report.get("source_subject_ids", []), "ingest_version": 1}
    if existing:
        existing.update(source_entry)
    else:
        proposed["source_files"].append(source_entry)
    validate_manifest(proposed)
    return proposed, accepted, pending


def _print_plan(manifest: dict, proposed: dict, accepted: list[dict], pending: list[dict], dry_run: bool) -> None:
    before = accepted_sessions(manifest)
    after = accepted_sessions(proposed)
    print(f"Manifest: {manifest['canonical_contract']['canonical_subject_name']} / {manifest['canonical_contract']['dataset_name']}\n")
    print("Current canonical:")
    print(f"  sessions = {len(before)}")
    print(f"  trials   = {sum(int(x['trial_count']) for x in before)}\n")
    print("Proposed:")
    for item in accepted:
        print(f"  accept {item['source_session_id']} -> {item['canonical_session_id']} ({item['trial_count']} trials)")
    for item in pending:
        print(f"  pending {item['source_session_id']} ({item['trial_count']} trials)")
    print("\nNew canonical:")
    print(f"  sessions = {len(after)}")
    print(f"  trials   = {sum(int(x['trial_count']) for x in after)}")
    if dry_run:
        print("\nDRY RUN — no files modified")


def apply(manifest_path: Path, manifest: dict, proposed: dict) -> None:
    canonical = _canonical_output(manifest_path, proposed)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{canonical.name}.", suffix=".tmp", dir=canonical.parent)
    os.close(fd)
    temporary = Path(temp_name)
    hashes_path = resolve_path(manifest_path, proposed.get("trial_hash_index", "trial_hashes.json"))
    try:
        validation = build_canonical(manifest_path=manifest_path, manifest=proposed, output=temporary)
        with h5py.File(temporary, "r") as handle:
            proposed["qc_reference"] = qc_reference(handle["data"][:], proposed["canonical_contract"]["channel_names"])
        proposed["canonical_build"] = {"version": 1, "validation": validation}
        hash_payload = rebuild_hash_index(manifest_path=manifest_path, manifest=proposed)
        validate_manifest(proposed)
        # All validation is complete before replacing either public artifact.
        os.replace(temporary, canonical)
        write_json_atomic(hashes_path, hash_payload)
        write_json_atomic(manifest_path, proposed)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    report_path = args.inspection_report.resolve()
    manifest = load_manifest(manifest_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    proposed, accepted, pending = propose(manifest_path, manifest, report_path, report, args)
    _print_plan(manifest, proposed, accepted, pending, not args.apply)
    if not args.apply:
        return
    apply(manifest_path, manifest, proposed)
    print("\nInspection validated")
    print("Manifest updated")
    print("Canonical rebuilt")
    print("Validation PASS")


if __name__ == "__main__":
    main()
