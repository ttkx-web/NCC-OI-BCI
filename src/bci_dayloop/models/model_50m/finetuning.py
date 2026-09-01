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


def resolve_backbone_adaptation(
    *,
    requested: str | None,
    unfreeze_last_n_blocks: int,
) -> str:
    """Resolve new adaptation CLI while preserving legacy partial commands."""
    if unfreeze_last_n_blocks < 0:
        raise ValueError("--unfreeze-last-n-blocks must be >= 0.")
    if requested is None:
        return "partial" if unfreeze_last_n_blocks > 0 else "frozen"
    if requested not in {"frozen", "partial", "lora"}:
        raise ValueError(f"Unsupported backbone adaptation: {requested!r}.")
    if requested == "frozen" and unfreeze_last_n_blocks != 0:
        raise ValueError(
            "--unfreeze-last-n-blocks must be 0 when "
            "--backbone-adaptation=frozen."
        )
    if requested == "partial" and unfreeze_last_n_blocks == 0:
        raise ValueError(
            "--backbone-adaptation=partial requires "
            "--unfreeze-last-n-blocks >= 1."
        )
    if requested == "lora" and unfreeze_last_n_blocks != 0:
        raise ValueError(
            "LoRA and partial fine-tuning are mutually exclusive; use "
            "--lora-last-n-blocks and leave --unfreeze-last-n-blocks at 0."
        )
    return requested
