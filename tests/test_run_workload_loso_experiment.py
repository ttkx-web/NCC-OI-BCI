from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_workload_loso_experiment.py"


def _module() -> object:
    scripts = str(SCRIPT.parent)
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location(
            "test_run_workload_loso_experiment_script", SCRIPT
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts)


def _plan(module: object, target: int = 1) -> object:
    return module.build_subject_plan(
        subjects=range(1, 16),
        target_subject=target,
        data_root=Path("data/processed/workload"),
        data_pattern="subject_{subject:02d}.h5",
        output_root=Path("experiments/workload_loso_full"),
        labram_checkpoint=Path("models/labram-base.pth"),
        cbramod_checkpoint=Path("models/cbramod.pth"),
        train_session="S1",
        validation_session="S2",
        final_test_session="S2",
        window_sec=2.0,
        device="cuda",
    )


def test_target_subject_is_excluded_from_population_subjects() -> None:
    module = _module()
    plan = _plan(module, target=7)

    assert plan.population_subjects == tuple(
        subject for subject in range(1, 16) if subject != 7
    )
    assert 7 not in plan.population_subjects


def test_full_workload_loso_expansion_has_isolated_subject_outputs() -> None:
    module = _module()
    args = module.build_parser().parse_args(
        [
            "--labram-checkpoint",
            "models/labram-base.pth",
            "--cbramod-checkpoint",
            "models/cbramod.pth",
            "--dry-run",
        ]
    )
    plans = module.build_plans(args)

    assert tuple(plan.target_subject for plan in plans) == tuple(range(1, 16))
    assert len({plan.output_dir for plan in plans}) == 15
    for plan in plans:
        target = plan.target_subject
        assert plan.output_dir == (
            Path("experiments/workload_loso_full") / f"subject_{target:02d}"
        )
        assert target not in plan.population_subjects
        assert len(plan.commands) == 6


def test_workload_default_sessions_separate_training_and_testing() -> None:
    module = _module()
    args = module.build_parser().parse_args(
        [
            "--subjects",
            "1",
            "2",
            "--labram-checkpoint",
            "labram.pth",
            "--cbramod-checkpoint",
            "cbramod.pth",
        ]
    )
    plan = module.build_plans(args)[0]
    commands = {command.name: command.argv for command in plan.commands}

    for train_name in ("train_labram", "train_cbramod"):
        command = commands[train_name]
        assert _arg_after(command, "--train-session") == "S1"
        assert _arg_after(command, "--validation-session") == "S2"
        assert _arg_after(command, "--final-test-session") == "S2"
        assert _arg_after(command, "--target-subject") == "1"
    for command_name in (
        "export_labram",
        "evaluate_labram",
        "export_cbramod",
        "evaluate_cbramod",
    ):
        assert _arg_after(commands[command_name], "--session") == "S2"

    cbramod_train = commands["train_cbramod"]
    labram_train = commands["train_labram"]
    assert _arg_after(cbramod_train, "--missing-channel-policy") == (
        "spherical_spline"
    )
    assert _arg_after(cbramod_train, "--min-observed-channels") == "21"
    assert "--missing-channel-policy" not in labram_train
    assert "--min-observed-channels" not in labram_train


def test_workload_session_cli_options_override_defaults() -> None:
    module = _module()
    args = module.build_parser().parse_args(
        [
            "--subjects",
            "1",
            "2",
            "--train-session",
            "TRAIN",
            "--validation-session",
            "VALIDATION",
            "--final-test-session",
            "TEST",
            "--labram-checkpoint",
            "labram.pth",
            "--cbramod-checkpoint",
            "cbramod.pth",
        ]
    )
    plan = module.build_plans(args)[0]
    commands = {command.name: command.argv for command in plan.commands}

    for train_name in ("train_labram", "train_cbramod"):
        command = commands[train_name]
        assert _arg_after(command, "--train-session") == "TRAIN"
        assert _arg_after(command, "--validation-session") == "VALIDATION"
        assert _arg_after(command, "--final-test-session") == "TEST"
    for command_name in (
        "export_labram",
        "evaluate_labram",
        "export_cbramod",
        "evaluate_cbramod",
    ):
        assert _arg_after(commands[command_name], "--session") == "TEST"


def test_dry_run_prints_commands_without_executing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()

    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run must not execute subprocesses")

    monkeypatch.setattr(module.subprocess, "run", unexpected_run)
    assert module.main(
        [
            "--subjects",
            "1",
            "2",
            "--labram-checkpoint",
            "labram.pth",
            "--cbramod-checkpoint",
            "cbramod.pth",
            "--dry-run",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "subject_01 population=2" in output
    assert "subject_02 population=1" in output
    assert "train_session=S1 validation_session=S2 final_test_session=S2" in output
    assert "[train_cbramod]" in output
    assert "--missing-channel-policy spherical_spline" in output
    assert "--min-observed-channels 21" in output
    assert "Dry-run complete: 2 subjects, no commands executed." in output


def _arg_after(argv: tuple[str, ...], flag: str) -> str:
    return argv[argv.index(flag) + 1]
