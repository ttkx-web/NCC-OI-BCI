from __future__ import annotations


def resolve_embedding_layer(
    *,
    requested: str,
    output_layer_idx: int,
    depth: int,
) -> int:
    """Resolve the user-facing 1-based downstream embedding layer."""
    if requested == "auto":
        resolved = int(output_layer_idx) + 1
    else:
        try:
            resolved = int(requested)
        except ValueError as error:
            raise ValueError(
                "--embedding-layer must be 'auto' or a 1-based integer, "
                f"got {requested!r}."
            ) from error

    if not 1 <= resolved <= depth:
        raise ValueError(
            "Embedding layer must be in the 1-based range "
            f"[1, {depth}], got {resolved}."
        )
    return resolved


def resolve_trainable_block_indices(
    *,
    embedding_layer: int,
    unfreeze_last_n_blocks: int,
) -> tuple[int, ...]:
    """Return 0-based blocks ending at the selected embedding layer."""
    if unfreeze_last_n_blocks < 0:
        raise ValueError("--unfreeze-last-n-blocks must be >= 0.")
    if unfreeze_last_n_blocks > embedding_layer:
        raise ValueError(
            "--unfreeze-last-n-blocks cannot exceed the selected "
            f"embedding layer: {unfreeze_last_n_blocks} > {embedding_layer}."
        )
    if unfreeze_last_n_blocks == 0:
        return ()
    return tuple(
        range(
            embedding_layer - unfreeze_last_n_blocks,
            embedding_layer,
        )
    )


def uses_frozen_feature_cache(*, unfreeze_last_n_blocks: int) -> bool:
    """Whether detached backbone feature caching is mathematically valid."""
    if unfreeze_last_n_blocks < 0:
        raise ValueError("--unfreeze-last-n-blocks must be >= 0.")
    return unfreeze_last_n_blocks == 0
