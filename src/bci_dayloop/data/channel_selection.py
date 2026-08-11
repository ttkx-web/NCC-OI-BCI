from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def strict_channel_indices(
    source_channel_names: Sequence[str],
    requested_channel_names: Sequence[str],
) -> tuple[int, ...]:
    """Resolve an explicit channel selection without aliases or filling."""

    source = tuple(str(name) for name in source_channel_names)
    requested = tuple(str(name) for name in requested_channel_names)

    if not source:
        raise ValueError("source_channel_names cannot be empty.")
    if not requested:
        raise ValueError("requested_channel_names cannot be empty.")
    if any(not name or name != name.strip() for name in source):
        raise ValueError(
            "source_channel_names contains an empty or untrimmed name."
        )
    if any(not name or name != name.strip() for name in requested):
        raise ValueError(
            "requested_channel_names contains an empty or untrimmed name."
        )

    duplicate_source = sorted(
        {name for name in source if source.count(name) > 1}
    )
    if duplicate_source:
        raise ValueError(
            "source_channel_names contains duplicates: "
            f"{duplicate_source}."
        )

    duplicate_requested = sorted(
        {name for name in requested if requested.count(name) > 1}
    )
    if duplicate_requested:
        raise ValueError(
            "requested_channel_names contains duplicates: "
            f"{duplicate_requested}."
        )

    source_index = {name: index for index, name in enumerate(source)}
    missing = [name for name in requested if name not in source_index]
    if missing:
        raise ValueError(
            "Requested channels are missing from the source: "
            f"{missing}."
        )

    return tuple(source_index[name] for name in requested)


def select_named_channels(
    data: np.ndarray,
    *,
    source_channel_names: Sequence[str],
    requested_channel_names: Sequence[str] | None,
    channel_axis: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Strictly select/reorder a named channel axis.

    ``None`` preserves the complete source order for backward compatibility.
    """

    array = np.asarray(data)
    source = tuple(str(name) for name in source_channel_names)
    requested = (
        source
        if requested_channel_names is None
        else tuple(str(name) for name in requested_channel_names)
    )
    axis = int(channel_axis)
    if axis < 0:
        axis += array.ndim
    if axis < 0 or axis >= array.ndim:
        raise ValueError(
            f"channel_axis {channel_axis} is invalid for shape {array.shape}."
        )
    if array.shape[axis] != len(source):
        raise ValueError(
            "Data channel dimension does not match source_channel_names: "
            f"{array.shape[axis]} != {len(source)}."
        )

    indices = strict_channel_indices(source, requested)
    selected = np.take(array, np.asarray(indices, dtype=np.int64), axis=axis)
    return selected, requested
