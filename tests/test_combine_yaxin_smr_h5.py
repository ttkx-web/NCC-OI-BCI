from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.combine_yaxin_smr_h5 import (
    CANONICAL_SESSION_IDS,
    CLASS_NAMES,
    DROP_CHANNELS,
    SOURCE_S01_SESSIONS,
    SOURCE_S02_SESSIONS,
    SourceFile,
    SourcePayload,
    build_combined_payload,
    filter_s02_auxiliary_channels,
)


def _balanced_labels(trials: int) -> np.ndarray:
    return np.repeat(np.arange(4, dtype=np.int64), trials // 4)


def _payload(
    *,
    file_id: int,
    subject_id: int,
    sessions: tuple[str, ...],
    channels: list[str],
    marker: float,
) -> SourcePayload:
    counts = [40 if index % 2 == 0 else 80 for index in range(len(sessions))]
    session_ids = np.concatenate([
        np.full(count, session, dtype=object) for session, count in zip(sessions, counts, strict=True)
    ])
    labels = np.concatenate([_balanced_labels(count) for count in counts])
    data = np.full((len(labels), len(channels), 3), marker, dtype=np.float32)
    return SourcePayload(
        source=SourceFile(Path(f"source_{file_id}.h5"), file_id, subject_id, sessions),
        data=data,
        labels=labels,
        subject_ids=np.full(len(labels), subject_id, dtype=np.int64),
        session_ids=session_ids,
        trial_ids=np.arange(len(labels), dtype=np.int64),
        channel_names=channels,
    )


def test_auxiliary_filter_is_name_based_and_preserves_shared_channel_order() -> None:
    s01_names = ["Fpz", "Fp1", "PO5"]
    s02_names = ["Fpz", "ECG", "Fp1", "HEOR", "PO5", "HEOL", "VEOU", "VEOL", "Trigger"]
    source = np.arange(2 * len(s02_names) * 3, dtype=np.float32).reshape(2, len(s02_names), 3)
    filtered, indices = filter_s02_auxiliary_channels(
        s01_channel_names=s01_names,
        s02_data=source,
        s02_channel_names=s02_names,
    )
    assert indices.tolist() == [0, 2, 4]
    assert np.array_equal(filtered, source[:, [0, 2, 4], :])
    assert tuple(DROP_CHANNELS) == ("ECG", "HEOR", "HEOL", "VEOU", "VEOL", "Trigger")


def test_combined_payload_canonicalizes_subject_sessions_labels_and_trial_ids() -> None:
    channels = ["Fpz", "Fp1", "PO5"]
    s01 = _payload(
        file_id=0, subject_id=1, sessions=SOURCE_S01_SESSIONS, channels=channels, marker=1.0
    )
    s02 = _payload(
        file_id=1,
        subject_id=2,
        sessions=SOURCE_S02_SESSIONS,
        channels=[*channels, *DROP_CHANNELS],
        marker=2.0,
    )
    payload = build_combined_payload(s01, s02)
    assert payload["data"].shape == (360, 3, 3)
    assert np.array_equal(payload["subject_ids"], np.ones(360, dtype=np.int64))
    assert np.array_equal(payload["trial_ids"], np.arange(360))
    assert np.array_equal(np.bincount(payload["labels"], minlength=4), np.full(4, 90))
    assert [str(value) for value in np.unique(payload["session_ids"])] == list(CANONICAL_SESSION_IDS)
    assert np.all(payload["data"][:120] == 1.0)
    assert np.all(payload["data"][120:] == 2.0)
    assert np.array_equal(payload["source_file_ids"][:120], np.zeros(120, dtype=np.int64))
    assert np.array_equal(payload["source_file_ids"][120:], np.ones(240, dtype=np.int64))
    assert CLASS_NAMES == ["left_hand", "right_hand", "both_hand", "rest"]

