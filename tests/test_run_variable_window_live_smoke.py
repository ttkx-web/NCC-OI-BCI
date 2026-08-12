from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import scripts.run_variable_window_live_smoke as smoke


@pytest.mark.parametrize(
    "command",
    (
        ("scripts/run_variable_window_live_smoke.py", "--help"),
        ("-m", "scripts.run_variable_window_live_smoke", "--help"),
    ),
)
def test_live_smoke_cli_help_supports_script_and_module_modes(
    command: tuple[str, ...],
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src"

    completed = subprocess.run(
        [sys.executable, *command],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--output-dir" in completed.stdout


def _combinations() -> tuple[smoke.SmokeCombination, ...]:
    return tuple(
        smoke.SmokeCombination(
            identifier=f"{model}_{seconds}s",
            model=model,
            window_seconds=float(seconds),
            package_relative_path=f"packages/{model}_{seconds}s",
            expected_prepared_shape=shape,
        )
        for model, shapes in (
            ("model_50m", ((1, 64, 100), (1, 64, 200), (1, 64, 300))),
            ("labram", ((1, 19, 1, 200), (1, 19, 2, 200), (1, 19, 3, 200))),
            ("cbramod", ((1, 22, 1, 200), (1, 22, 2, 200), (1, 22, 3, 200))),
        )
        for seconds, shape in enumerate(shapes, start=1)
    )


def _probe_summary(combination: smoke.SmokeCombination) -> dict[str, object]:
    values: dict[str, object] = {
        "status": "passed",
        "model_type": combination.model,
        "compatibility_status": "passed",
        "is_test_head": False,
        "prepared_shape": list(combination.expected_prepared_shape),
        "received_packets": 20,
        "received_samples": 1000,
        "emitted_windows": 2,
        "model_input_safe_count": 2,
        "prediction_success_count": 2,
        "failed_windows": 0,
        "pipeline_failed_windows": 0,
        "model_input_failure_count": 0,
        "prediction_failure_count": 0,
        "missing_packets": 0,
        "duplicate_packets": 0,
        "out_of_order_packets": 0,
        "gap_count": 0,
        "waveforms_saved": False,
        "last_error": None,
        "pre_disconnect_health": {
            "malformed_packets": 0,
            "reconnect_count": 0,
        },
    }
    if combination.model == "cbramod":
        values.update(
            {
                "observed_channel_count": 19,
                "missing_channel_names": ["CPz", "P1", "P2"],
                "completion_policy": "spherical_spline",
                "completion_matrix_sha256": "completion-sha",
            }
        )
    return values


def _install_packages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[smoke.SmokeCombination, ...]:
    combinations = _combinations()
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    monkeypatch.setattr(smoke, "SMOKE_COMBINATIONS", combinations)
    for combination in combinations:
        package = tmp_path / combination.package_relative_path
        package.mkdir(parents=True)
        if combination.model == "cbramod":
            (package / "package.yaml").write_text(
                "runtime:\n"
                "  channel_completion:\n"
                "    completion_matrix_sha256: completion-sha\n",
                encoding="utf-8",
            )
    return combinations


def test_orchestrator_runs_all_nine_sequentially_and_writes_sanitized_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    combinations = _install_packages(monkeypatch, tmp_path)
    monkeypatch.setenv(smoke.HOST_ENV, "sensitive-neuracle-host")
    calls: list[tuple[str, ...]] = []

    def fake_probe(argv: list[str]) -> int:
        calls.append(tuple(argv))
        package = Path(argv[argv.index("--package") + 1])
        combination = next(
            item for item in combinations
            if package == tmp_path / item.package_relative_path
        )
        assert argv[argv.index("--device") + 1] == "cuda"
        assert argv[argv.index("--duration-sec") + 1] == "20.0"
        assert argv[argv.index("--step-sec") + 1] == "0.5"
        assert argv[argv.index("--window-sec") + 1] == str(combination.window_seconds)
        summary_dir = Path(argv[argv.index("--output-dir") + 1])
        summary_dir.mkdir(parents=True)
        (summary_dir / "runtime_inference_summary.json").write_text(
            json.dumps(_probe_summary(combination)), encoding="utf-8"
        )
        return 0

    monkeypatch.setattr(smoke, "run_runtime_probe", fake_probe)
    output_dir = tmp_path / "out"

    assert smoke.main(["--output-dir", str(output_dir)]) == 0
    assert len(calls) == 9
    assert [call[call.index("--package") + 1] for call in calls] == [
        str(tmp_path / item.package_relative_path) for item in combinations
    ]
    report = json.loads(
        (output_dir / "variable_window_smoke_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["combination_count"] == report["passed_count"] == 9
    assert report["all_passed"] is True
    assert "sensitive-neuracle-host" not in json.dumps(report)
    assert (output_dir / "variable_window_smoke_summary.csv").is_file()
    assert (output_dir / "variable_window_smoke_summary.md").is_file()


def test_orchestrator_records_a_failure_and_continues_remaining_combinations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    combinations = _install_packages(monkeypatch, tmp_path)
    monkeypatch.setenv(smoke.HOST_ENV, "sensitive-neuracle-host")
    calls: list[str] = []

    def fake_probe(argv: list[str]) -> int:
        package = Path(argv[argv.index("--package") + 1])
        combination = next(
            item for item in combinations
            if package == tmp_path / item.package_relative_path
        )
        calls.append(combination.identifier)
        payload = _probe_summary(combination)
        if combination.identifier == "labram_2s":
            payload["prediction_failure_count"] = 1
            payload["status"] = "failed"
            payload["last_error"] = "runtime_prediction_failed"
        summary_dir = Path(argv[argv.index("--output-dir") + 1])
        summary_dir.mkdir(parents=True)
        (summary_dir / "runtime_inference_summary.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return 2 if combination.identifier == "labram_2s" else 0

    monkeypatch.setattr(smoke, "run_runtime_probe", fake_probe)
    output_dir = tmp_path / "out"

    assert smoke.main(["--output-dir", str(output_dir)]) == 2
    assert calls == [item.identifier for item in combinations]
    report = json.loads(
        (output_dir / "variable_window_smoke_summary.json").read_text(
            encoding="utf-8"
        )
    )
    failed = next(
        row for row in report["combinations"] if row["model"] == "labram" and row["window_sec"] == 2.0
    )
    assert failed["status"] == "failed"
    assert "prediction_failure_count" in failed["failure_reasons"]
    assert "probe_exit_code" in failed["failure_reasons"]


def test_orchestrator_requires_environment_host_and_fixed_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.delenv(smoke.HOST_ENV, raising=False)
    with pytest.raises(ValueError, match="environment"):
        smoke.main(["--output-dir", str(tmp_path / "out")])

    monkeypatch.setenv(smoke.HOST_ENV, "sensitive-neuracle-host")
    with pytest.raises(ValueError, match="20 seconds"):
        smoke.main(["--duration-sec", "10"])
