from __future__ import annotations

import numpy as np

from bci_dayloop.demo.visual_state import VisualState


def test_visual_state_uses_wall_time_and_interpolates_decode_targets() -> None:
    state = VisualState()
    state.advance(10.0, playback_speed=1.0)
    state.set_decode_targets(
        np.asarray([1.0, 3.0]),
        np.full((2, 2, 4), 20, dtype=np.uint8),
        np.full((2, 2, 4), 30, dtype=np.uint8),
    )
    np.testing.assert_allclose(state.displayed_psd, [1.0, 3.0])

    state.advance(10.05, playback_speed=2.0)
    state.set_decode_targets(
        np.asarray([5.0, 7.0]),
        np.full((2, 2, 4), 100, dtype=np.uint8),
        np.full((2, 2, 4), 120, dtype=np.uint8),
    )
    state.advance(10.10, playback_speed=2.0)
    state.interpolate(decode_interval_sec=0.2)

    assert 1.0 < state.displayed_psd[0] < 5.0
    assert 3.0 < state.displayed_psd[1] < 7.0
    np.testing.assert_array_equal(state.displayed_cortical_rgba("left"), np.full((2, 2, 4), 100, dtype=np.uint8))
    np.testing.assert_array_equal(state.displayed_cortical_rgba("right"), np.full((2, 2, 4), 120, dtype=np.uint8))
    assert state.median_visual_fps is not None


def test_visual_state_pause_and_reset_freeze_and_clear_visual_interpolation() -> None:
    state = VisualState()
    state.advance(1.0, playback_speed=1.0)
    state.advance(1.1, playback_speed=1.0)
    frozen_time = state.stream_time_sec
    state.pause()
    state.advance(3.0, playback_speed=1.0)
    assert state.stream_time_sec == frozen_time

    state.reset()
    assert state.stream_time_sec == 0.0
    assert state.displayed_psd is None
    assert state.displayed_cortical_left is None
