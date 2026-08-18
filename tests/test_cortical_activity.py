from __future__ import annotations

import inspect

import numpy as np
import pytest

from bci_dayloop.demo.cortical_activity import CorticalActivityMapper
from bci_dayloop.demo.cortical_montage import load_cortical_montage, validate_cortical_montage
from bci_dayloop.demo.signal_features import extract_signal_features


def test_cortical_mapper_uses_name_matched_cached_montage_masks() -> None:
    mapper = CorticalActivityMapper()
    masks_id = id(mapper._masks)
    channels = ["FC3", "FC4", "C3", "C4", "CP3", "CP4"]
    power = np.asarray([1.0, 1.5, 5.0, 1.8, 1.1, 1.2])

    result = mapper.update(channels, power, montage_name="bnci_22")

    assert id(mapper._masks) == masks_id
    assert result.available
    assert result.mapped_channel_count == len(channels)
    assert result.left_rgba.shape == mapper.left_template.shape
    assert result.right_rgba.shape == mapper.right_template.shape
    assert result.left_rgba[..., 3].min() == 0
    assert result.right_rgba[..., 3].min() == 0
    assert result.update_ms < 1000.0


def test_invalid_and_unknown_channels_do_not_contribute_to_cortical_normalization() -> None:
    mapper = CorticalActivityMapper(ema_alpha=0.0)
    result = mapper.update(
        ["FC3", "FC4", "UNKNOWN_1"],
        np.asarray([1.0, 1e9, 1e9]),
        channel_valid_mask=np.asarray([True, False, True]),
    )
    assert result.available
    assert result.mapped_channel_count == 1
    assert result.unmapped_channel_count == 1

    fallback = mapper.update(["UNKNOWN_1"], np.asarray([1.0]))
    assert not fallback.available
    assert fallback.mapped_channel_count == 0


def test_channel_order_permutation_preserves_named_cortical_projection() -> None:
    names = ["FC3", "FC4", "C3", "C4"]
    power = np.asarray([1.0, 2.0, 6.0, 3.0])
    first = CorticalActivityMapper(ema_alpha=0.0).update(names, power)
    order = np.asarray([2, 0, 3, 1])
    second = CorticalActivityMapper(ema_alpha=0.0).update([names[index] for index in order], power[order])
    np.testing.assert_array_equal(first.left_rgba, second.left_rgba)
    np.testing.assert_array_equal(first.right_rgba, second.right_rgba)


def test_1_to_30_hz_absolute_power_excludes_strong_40_hz_component_and_is_unit_invariant() -> None:
    sample_rate = 250.0
    time = np.arange(int(sample_rate * 4.0)) / sample_rate
    ten_hz = 40.0 * np.sin(2 * np.pi * 10.0 * time)
    forty_hz = 250.0 * np.sin(2 * np.pi * 40.0 * time)
    uv = np.vstack((ten_hz, forty_hz))
    features_uv = extract_signal_features(uv, sample_rate, unit="uV", channel_names=["FC3", "FC4"])
    features_v = extract_signal_features(uv * 1e-6, sample_rate, unit="V", channel_names=["FC3", "FC4"])

    assert features_uv.channel_power_1_30[0] > features_uv.channel_power_1_30[1] * 20.0
    np.testing.assert_allclose(features_uv.channel_power_1_30, features_v.channel_power_1_30, rtol=2e-5, atol=1e-8)


def test_stronger_1_to_30_power_creates_stronger_named_hotspot_and_ema_is_decoder_rate() -> None:
    low = np.asarray([1.0, 1.0])
    high_left = np.asarray([10.0, 1.0])
    mapper = CorticalActivityMapper(ema_alpha=0.85)
    mapper.update(["FC3", "FC4"], low)
    previous_left = mapper._previous_heatmaps["left"].copy()
    mapper.update(["FC3", "FC4"], high_left)
    smoothed_left = mapper._previous_heatmaps["left"]

    instantaneous = CorticalActivityMapper(ema_alpha=0.85)
    instantaneous.update(["FC3", "FC4"], high_left)
    target_left = instantaneous._previous_heatmaps["left"]
    np.testing.assert_allclose(smoothed_left, 0.85 * previous_left + 0.15 * target_left)
    assert float(target_left.max()) > 0.0
    assert float(instantaneous._previous_heatmaps["right"].max()) == 0.0


def test_montage_registry_exposes_explicit_32_channel_placeholder() -> None:
    montage = load_cortical_montage("bcigo_32_placeholder")
    assert montage.device_name == "BCIGo"
    assert len(montage.channels) == 32


def test_montage_validation_rejects_invalid_anchor_before_decode() -> None:
    with pytest.raises(ValueError, match="x/y"):
        validate_cortical_montage(
            {
                "device_name": "test",
                "montage_name": "test_montage",
                "channels": {"C3": {"anchors": [{"hemisphere": "left", "x": 1.2, "y": 0.5}]}},
            }
        )


def test_cortical_runtime_has_no_nilearn_dependency() -> None:
    import bci_dayloop.demo.cortical_activity as cortical_activity

    assert "nilearn" not in inspect.getsource(cortical_activity).lower()
