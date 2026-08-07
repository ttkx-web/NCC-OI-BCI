from __future__ import annotations

import pytest

from bci_dayloop.models.model_50m.config import STANDARD_64_CHANNELS
from bci_dayloop.realtime.runtime_mapping import (
    APPROVED_NEURACLE_59_TO_STANDARD64,
    ApprovedRealtimeMappingPolicy,
)


def test_approved_policy_exactly_captures_the_authorized_mapping() -> None:
    policy = APPROVED_NEURACLE_59_TO_STANDARD64

    assert policy.policy_id == "neuracle_59_to_standard64_v1"
    assert policy.target_channel_names is STANDARD_64_CHANNELS
    assert len(policy.source_channel_names) == 59
    assert len(policy.target_channel_names) == 64
    assert len(policy.exact_matched_channels) == 57
    assert policy.missing_target_channels == ("AFz", "CPz", "P1", "P2", "Iz", "F9", "F10")
    assert policy.ignored_source_channels == ("PO5", "PO6")
    assert policy.target_missing_indices == (5, 39, 47, 49, 61, 62, 63)
    assert policy.expected_valid_channels == 57
    assert policy.allow_zero_fill_missing is True
    assert policy.require_channel_valid_mask is True


def test_policy_fails_closed_if_runtime_standard_channel_definition_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bci_dayloop.realtime.runtime_mapping as mapping

    changed_target = STANDARD_64_CHANNELS[:-1] + ("A2",)
    monkeypatch.setattr(mapping, "STANDARD_64_CHANNELS", changed_target)

    with pytest.raises(ValueError, match="STANDARD_64_CHANNELS changed"):
        ApprovedRealtimeMappingPolicy(target_channel_names=changed_target)
