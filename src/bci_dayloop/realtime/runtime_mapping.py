"""Approved, immutable mapping policy for the Stage 2B realtime source."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from bci_dayloop.models.model_50m.config import STANDARD_64_CHANNELS


_SOURCE_CHANNEL_NAMES: tuple[str, ...] = (
    "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8",
    "Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
    "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FT7", "FT8",
    "Cz", "C1", "C2", "C3", "C4", "C5", "C6", "T7", "T8",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "TP7", "TP8",
    "Pz", "P3", "P4", "P5", "P6", "P7", "P8",
    "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8", "Oz", "O1", "O2",
)
_EXPECTED_MISSING_TARGET_CHANNELS = ("AFz", "CPz", "P1", "P2", "Iz", "F9", "F10")
_EXPECTED_IGNORED_SOURCE_CHANNELS = ("PO5", "PO6")
_STANDARD64_FINGERPRINT = "de977434c379296649a3237f5f0f769a0e8e48fcb6e04505edab89fe8db9b38c"


@dataclass(frozen=True, slots=True)
class ApprovedRealtimeMappingPolicy:
    """The sole approved 59-to-64 policy; target names reference Runtime code."""

    policy_id: str = "neuracle_59_to_standard64_v1"
    source_channel_names: tuple[str, ...] = _SOURCE_CHANNEL_NAMES
    target_channel_names: tuple[str, ...] = STANDARD_64_CHANNELS
    expected_valid_channels: int = 57
    expected_missing_target_channels: tuple[str, ...] = _EXPECTED_MISSING_TARGET_CHANNELS
    expected_ignored_source_channels: tuple[str, ...] = _EXPECTED_IGNORED_SOURCE_CHANNELS
    allow_zero_fill_missing: bool = True
    require_channel_valid_mask: bool = True

    def __post_init__(self) -> None:
        if self.policy_id != "neuracle_59_to_standard64_v1":
            raise ValueError("unexpected realtime mapping policy_id")
        if self.source_channel_names != _SOURCE_CHANNEL_NAMES:
            raise ValueError("policy source must be the approved 59-channel order")
        if self.target_channel_names != STANDARD_64_CHANNELS:
            raise ValueError("policy target must be STANDARD_64_CHANNELS")
        fingerprint = hashlib.sha256("\x1f".join(self.target_channel_names).encode("utf-8")).hexdigest()
        if fingerprint != _STANDARD64_FINGERPRINT:
            raise ValueError("STANDARD_64_CHANNELS changed; review the approved policy")
        if len(self.source_channel_names) != 59 or len(self.target_channel_names) != 64:
            raise ValueError("approved policy channel counts changed")
        if len(set(self.source_channel_names)) != 59 or len(set(self.target_channel_names)) != 64:
            raise ValueError("approved policy channel names must be unique")
        if len(self.exact_matched_channels) != self.expected_valid_channels:
            raise ValueError("approved exact matched channel count changed")
        if self.expected_valid_channels != 57:
            raise ValueError("approved expected valid channel count changed")
        if self.expected_missing_target_channels != _EXPECTED_MISSING_TARGET_CHANNELS:
            raise ValueError("policy expected missing target channels changed")
        if self.expected_ignored_source_channels != _EXPECTED_IGNORED_SOURCE_CHANNELS:
            raise ValueError("policy expected ignored source channels changed")
        if self.missing_target_channels != self.expected_missing_target_channels:
            raise ValueError("approved missing target channels changed")
        if self.ignored_source_channels != self.expected_ignored_source_channels:
            raise ValueError("approved ignored source channels changed")
        if self.allow_zero_fill_missing is not True or self.require_channel_valid_mask is not True:
            raise ValueError("approved policy requires explicit zero-fill and channel_valid_mask")

    @property
    def exact_matched_channels(self) -> tuple[str, ...]:
        return tuple(name for name in self.source_channel_names if name in self.target_channel_names)

    @property
    def missing_target_channels(self) -> tuple[str, ...]:
        return tuple(name for name in self.target_channel_names if name not in self.source_channel_names)

    @property
    def ignored_source_channels(self) -> tuple[str, ...]:
        return tuple(name for name in self.source_channel_names if name not in self.target_channel_names)

    @property
    def target_missing_indices(self) -> tuple[int, ...]:
        return tuple(self.target_channel_names.index(name) for name in self.expected_missing_target_channels)


APPROVED_NEURACLE_59_TO_STANDARD64 = ApprovedRealtimeMappingPolicy()
