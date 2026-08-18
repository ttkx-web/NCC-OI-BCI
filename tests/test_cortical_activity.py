from __future__ import annotations

import inspect

import numpy as np

from bci_dayloop.demo.cortical_activity import CorticalActivityMapper


def test_cortical_mapper_reuses_static_masks_and_limits_output_to_template() -> None:
    mapper = CorticalActivityMapper()
    masks_id = id(mapper._masks)
    channels = ["F3", "F4", "C3", "C4", "P3", "P4"]
    powers = {
        "theta": np.asarray([0.08, 0.09, 0.12, 0.07, 0.05, 0.06]),
        "alpha": np.asarray([0.25, 0.19, 0.31, 0.18, 0.12, 0.11]),
        "beta": np.asarray([0.11, 0.14, 0.16, 0.10, 0.07, 0.08]),
    }

    result = mapper.update(channels, powers)

    assert id(mapper._masks) == masks_id
    assert result.left_rgba.shape == mapper.left_template.shape
    assert result.right_rgba.shape == mapper.right_template.shape
    assert result.left_rgba[..., 3].min() == 0
    assert result.right_rgba[..., 3].min() == 0
    assert result.update_ms < 1000.0


def test_cortical_runtime_has_no_nilearn_dependency() -> None:
    import bci_dayloop.demo.cortical_activity as cortical_activity

    assert "nilearn" not in inspect.getsource(cortical_activity).lower()
