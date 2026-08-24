from __future__ import annotations

import sys
from pathlib import Path

import pytest

from bci_dayloop.applications.three_mental_states.export_config import (
    DEFAULT_EXPORT_CONFIG_PATH,
    load_three_mental_state_export_config,
)
from bci_dayloop.utils.config import dump_yaml, load_yaml, project_root


def _payload() -> dict[str, object]:
    return load_yaml(project_root() / DEFAULT_EXPORT_CONFIG_PATH)


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "export.yaml"
    dump_yaml(payload, path)
    return path


def test_default_export_config_loads_and_resolves_from_repository_root() -> None:
    config = load_three_mental_state_export_config()
    assert config.sources.backbone_checkpoint == project_root() / "checkpoints/backbones/50m/model_deploy.pt"
    assert tuple(config.sources.heads) == ("workload", "attention", "emotion")
    assert config.runtime.step_sec == config.runtime.window_sec == 2.0


@pytest.mark.parametrize("mutation", [
    lambda payload: payload["sources"]["heads"].pop("workload"),  # type: ignore[index]
    lambda payload: payload["sources"]["heads"].update({"unknown": "head.pt"}),  # type: ignore[index]
    lambda payload: payload.update({"schema_version": 2}),
    lambda payload: payload["runtime"].update({"target_sample_rate_hz": 0}),  # type: ignore[index]
    lambda payload: payload["runtime"].update({"step_sec": 3.0}),  # type: ignore[index]
])
def test_invalid_export_config_fails_fast(tmp_path: Path, mutation) -> None:
    payload = _payload(); mutation(payload)
    with pytest.raises(ValueError):
        load_three_mental_state_export_config(_write(tmp_path, payload))


def test_export_cli_overrides_yaml_without_argparse_default_ambiguity(tmp_path: Path) -> None:
    scripts = project_root() / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        import export_50m_multi_head_model_package as export_script
        config_path = _write(tmp_path, _payload())
        resolved = export_script.resolved_export_arguments(export_script.build_parser().parse_args([
            "--config", str(config_path), "--package-version", "2", "--output-dir", "custom-package",
        ]))
        assert resolved["package_version"] == "2"
        assert resolved["output_dir"] == project_root() / "custom-package"
        legacy = export_script.build_parser().parse_args([
            "--backbone-checkpoint", "backbone.pt", "--workload-head", "workload.pt",
            "--attention-head", "attention.pt", "--emotion-head", "emotion.pt", "--output-dir", "out",
        ])
        assert legacy.backbone_checkpoint == "backbone.pt"
    finally:
        sys.path.remove(str(scripts))
