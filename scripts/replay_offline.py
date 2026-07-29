from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import ROOT  # noqa: F401
from bci_dayloop.acquisition.factory import AcquirerFactory
from bci_dayloop.data.hdf5_dataset import EEGHDF5, HDF5Metadata
from bci_dayloop.data.preprocessing import EEGPreprocessor
from bci_dayloop.inference.observability import (
    JsonlWindowLogger,
    PipelineRunStats,
    PipelineStatsSnapshot,
    calculate_expected_windows,
)
from bci_dayloop.inference.realtime import SlidingWindowDecoder
from bci_dayloop.inference.run_report import PipelineRunReport
from bci_dayloop.inference.runtime_control import PipelineController, PipelineControllerSnapshot, PipelineState
from bci_dayloop.models.factory import ModelFactory
from bci_dayloop.utils.config import load_yaml, resolve_path


@dataclass(frozen=True, slots=True)
class ReplaySettings:
    data_path: Path
    model_package: Path
    device: str
    acquirer_name: str
    session: str
    loop: bool
    window_sec: float
    step_sec: float
    replay_speed: float
    confidence_threshold: float
    maximum_windows: int | None
    jsonl_log_path: Path | None
    summary_json_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pseudo-realtime HDF5 replay through a model package")
    parser.add_argument("--config", default="configs/day1_bnci_s01.yaml")
    parser.add_argument("--data")
    parser.add_argument("--model-package")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--window-sec", type=float)
    parser.add_argument("--step-sec", type=float)
    parser.add_argument("--replay-speed", type=float)
    jsonl_group = parser.add_mutually_exclusive_group()
    jsonl_group.add_argument("--jsonl-log")
    jsonl_group.add_argument("--no-jsonl-log", action="store_true")
    parser.add_argument("--summary-json")
    return parser


def _first_defined(command_line: Any, yaml_value: Any, default: Any) -> Any:
    return command_line if command_line is not None else yaml_value if yaml_value is not None else default


def resolve_replay_settings(args: argparse.Namespace, config: dict[str, Any]) -> ReplaySettings:
    replay = dict(config.get("replay", {}))
    model_config = dict(config.get("model", {}))
    project = dict(config.get("project", {}))
    data_config = dict(config.get("data", {}))
    run_dir = Path(project.get("run_dir", "runs/day1_bnci_s01"))

    data_value = _first_defined(args.data, data_config.get("output_hdf5"), "data/processed/bnci2014_001_s01.h5")
    package_value = _first_defined(args.model_package, replay.get("model_package"), run_dir / "model_package")
    device = str(_first_defined(args.device, model_config.get("device"), "cuda"))
    maximum_value = _first_defined(args.max_windows, replay.get("max_windows"), 100)
    maximum_windows = None if maximum_value is None else int(maximum_value)
    if maximum_windows is not None and maximum_windows < 0:
        raise ValueError("max_windows must be non-negative or omitted")

    window_sec = float(_first_defined(args.window_sec, replay.get("window_sec"), 4.0))
    step_sec = float(_first_defined(args.step_sec, replay.get("step_sec"), 0.5))
    replay_speed = float(_first_defined(args.replay_speed, replay.get("speed"), 1.0))
    if window_sec <= 0:
        raise ValueError("window_sec must be greater than zero")
    if step_sec <= 0:
        raise ValueError("step_sec must be greater than zero")
    if step_sec > window_sec:
        raise ValueError("step_sec must not be greater than window_sec")
    if replay_speed <= 0:
        raise ValueError("replay_speed must be greater than zero")

    if args.no_jsonl_log:
        jsonl_log_path = None
    else:
        jsonl_value = _first_defined(args.jsonl_log, replay.get("jsonl_log"), None)
        jsonl_log_path = resolve_path(jsonl_value) if jsonl_value is not None else None
    summary_value = _first_defined(args.summary_json, replay.get("summary_json"), run_dir / "replay_summary.json")

    return ReplaySettings(
        data_path=resolve_path(data_value),
        model_package=resolve_path(package_value),
        device=device,
        acquirer_name=str(replay.get("acquirer", "replay")),
        session=str(replay.get("session", "1test")),
        loop=bool(replay.get("loop", False)),
        window_sec=window_sec,
        step_sec=step_sec,
        replay_speed=replay_speed,
        confidence_threshold=float(replay.get("confidence_threshold", 0.55)),
        maximum_windows=maximum_windows,
        jsonl_log_path=jsonl_log_path,
        summary_json_path=resolve_path(summary_value),
    )


def expected_and_target_windows(
    *,
    trial_count: int,
    samples_per_trial: int,
    sample_rate: float,
    window_sec: float,
    step_sec: float,
    maximum_windows: int | None,
) -> tuple[int, int]:
    if trial_count < 0 or samples_per_trial < 0:
        raise ValueError("trial_count and samples_per_trial must be non-negative")
    window_samples = round(window_sec * sample_rate)
    step_samples = round(step_sec * sample_rate)
    expected = calculate_expected_windows(trial_count * samples_per_trial, window_samples, step_samples)
    target = expected if maximum_windows is None else min(expected, maximum_windows)
    return expected, target


def build_report(
    settings: ReplaySettings,
    *,
    model_name: str | None,
    metadata: HDF5Metadata | None,
    expected_windows: int,
    target_windows: int,
    stats_snapshot: PipelineStatsSnapshot,
    controller_snapshot: PipelineControllerSnapshot | None,
    fallback_state: PipelineState = PipelineState.IDLE,
    fallback_error: Exception | None = None,
) -> PipelineRunReport:
    if controller_snapshot is None:
        state = fallback_state.value
        run_id = 0
        error_type = type(fallback_error).__name__ if fallback_error else None
        error_message = str(fallback_error) if fallback_error else None
    else:
        state = controller_snapshot.state.value
        run_id = controller_snapshot.run_id
        error_type = controller_snapshot.last_error_type
        error_message = controller_snapshot.last_error_message
        if error_type is None and fallback_error is not None:
            error_type = type(fallback_error).__name__
            error_message = str(fallback_error)
    return PipelineRunReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        controller_state=state,
        run_id=run_id,
        model_package=str(settings.model_package),
        model_name=model_name,
        device=settings.device,
        data_path=str(settings.data_path),
        session=settings.session,
        sample_rate=metadata.sample_rate if metadata is not None else None,
        input_unit=metadata.unit if metadata is not None else None,
        window_sec=settings.window_sec,
        step_sec=settings.step_sec,
        replay_speed=settings.replay_speed,
        maximum_windows=settings.maximum_windows,
        expected_windows=expected_windows,
        target_windows=target_windows,
        emitted_windows=stats_snapshot.emitted_windows,
        successful_windows=stats_snapshot.successful_windows,
        failed_windows=stats_snapshot.failed_windows,
        chunks_received=stats_snapshot.chunks_received,
        runtime_sec=stats_snapshot.runtime_sec,
        current_latency_ms=stats_snapshot.current_latency_ms,
        average_latency_ms=stats_snapshot.average_latency_ms,
        p95_latency_ms=stats_snapshot.p95_latency_ms,
        preprocessing_average_ms=stats_snapshot.preprocessing_average_ms,
        model_average_ms=stats_snapshot.model_average_ms,
        jsonl_log_path=str(settings.jsonl_log_path) if settings.jsonl_log_path is not None else None,
        last_error_type=error_type,
        last_error_message=error_message,
    )


def build_pipeline_controller(
    settings: ReplaySettings,
    *,
    model: Any,
    preprocessor: EEGPreprocessor,
    metadata: HDF5Metadata,
    command_map: dict[str, str],
    stats: PipelineRunStats,
    target_windows: int,
) -> PipelineController:
    logger = JsonlWindowLogger(settings.jsonl_log_path) if settings.jsonl_log_path is not None else None
    decoder = SlidingWindowDecoder(
        model,
        preprocessor,
        metadata.class_names,
        sample_rate=metadata.sample_rate,
        input_unit=metadata.unit,
        window_sec=settings.window_sec,
        step_sec=settings.step_sec,
        confidence_threshold=settings.confidence_threshold,
        command_map=command_map,
        run_stats=stats,
        jsonl_logger=logger,
    )

    def acquirer_factory():
        return AcquirerFactory.create(
            settings.acquirer_name,
            data_path=settings.data_path,
            session=settings.session,
            speed=settings.replay_speed,
            loop=settings.loop,
            window_sec=settings.window_sec,
            step_sec=settings.step_sec,
        )

    return PipelineController(decoder, acquirer_factory, max_windows=target_windows)


def stop_after_keyboard_interrupt(controller: PipelineController | None) -> Exception | None:
    if controller is None:
        return None
    try:
        controller.stop(wait=True, timeout=5.0)
    except Exception as error:  # noqa: BLE001
        return error
    return None


def print_summary(report: PipelineRunReport, summary_path: Path) -> None:
    def format_ms(value: float | None) -> str:
        return "—" if value is None else f"{value:.1f} ms"

    print(f"State: {report.controller_state}")
    print(f"Expected / target / emitted windows: {report.expected_windows} / {report.target_windows} / {report.emitted_windows}")
    print(f"Successful / failed windows: {report.successful_windows} / {report.failed_windows}")
    print(f"Runtime: {report.runtime_sec:.3f} s")
    print(
        "Current / average / P95 latency: "
        f"{format_ms(report.current_latency_ms)} / {format_ms(report.average_latency_ms)} / {format_ms(report.p95_latency_ms)}"
    )
    print(f"Preprocessing average: {format_ms(report.preprocessing_average_ms)}")
    print(f"Model average: {format_ms(report.model_average_ms)}")
    print(f"JSONL path: {report.jsonl_log_path or 'disabled'}")
    print(f"Summary JSON path: {summary_path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_yaml(resolve_path(args.config))
    settings = resolve_replay_settings(args, config)
    stats = PipelineRunStats()
    metadata: HDF5Metadata | None = None
    model_name: str | None = None
    expected_windows = 0
    target_windows = 0
    controller: PipelineController | None = None
    caught_error: Exception | None = None
    interrupted = False

    try:
        dataset = EEGHDF5(settings.data_path)
        metadata = dataset.metadata
        trials = dataset.load(settings.session)["data"]
        expected_windows, target_windows = expected_and_target_windows(
            trial_count=int(trials.shape[0]),
            samples_per_trial=int(trials.shape[-1]),
            sample_rate=metadata.sample_rate,
            window_sec=settings.window_sec,
            step_sec=settings.step_sec,
            maximum_windows=settings.maximum_windows,
        )
        stats.set_expected_windows(target_windows)

        model_yaml = load_yaml(settings.model_package / "model.yaml")
        model_name = str(model_yaml.get("name")) if model_yaml.get("name") is not None else None
        model = ModelFactory.load_package(settings.model_package, device=settings.device)
        preprocessor = EEGPreprocessor(load_yaml(settings.model_package / "preprocessing.yaml"))
        with (settings.model_package / "command_map.json").open("r", encoding="utf-8") as handle:
            command_map = json.load(handle)
        controller = build_pipeline_controller(
            settings,
            model=model,
            preprocessor=preprocessor,
            metadata=metadata,
            command_map=command_map,
            stats=stats,
            target_windows=target_windows,
        )
        controller.start()
        controller.wait()
        controller.raise_if_failed()
    except KeyboardInterrupt:
        interrupted = True
        caught_error = stop_after_keyboard_interrupt(controller)
    except Exception as error:  # noqa: BLE001
        caught_error = error
    finally:
        snapshot = controller.snapshot() if controller is not None else None
        fallback_state = PipelineState.STOPPED if interrupted and caught_error is None else PipelineState.FAILED
        report = build_report(
            settings,
            model_name=model_name,
            metadata=metadata,
            expected_windows=expected_windows,
            target_windows=target_windows,
            stats_snapshot=stats.snapshot(),
            controller_snapshot=snapshot,
            fallback_state=fallback_state,
            fallback_error=caught_error,
        )
        try:
            report.save_json(settings.summary_json_path)
        except Exception as summary_error:  # noqa: BLE001
            print(f"Failed to write summary JSON: {summary_error}", file=sys.stderr)
            if caught_error is None:
                caught_error = summary_error
        print_summary(report, settings.summary_json_path)

    if interrupted:
        if caught_error is not None:
            print(f"Interrupted; pipeline stop failed: {caught_error}", file=sys.stderr)
            return 1
        return 130
    if caught_error is not None:
        print(f"Pipeline failed: {type(caught_error).__name__}: {caught_error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
