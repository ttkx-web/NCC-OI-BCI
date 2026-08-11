from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _bootstrap import ROOT  # noqa: F401

from bci_dayloop.benchmarking.core import (
    RuntimeBenchmarkCore,
)
from bci_dayloop.benchmarking.reporting import (
    BenchmarkCandidate,
    build_candidate_summary,
    flatten_window_record,
    write_benchmark_summary_json,
    write_window_records_csv,
)
from bci_dayloop.benchmarking.windows import (
    ReplayWindowProvider,
)
from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.packages.loader import (
    LoadedRuntimePackage,
    load_runtime_package,
)
from bci_dayloop.utils.config import (
    dump_json,
    load_yaml,
    resolve_path,
)


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate_id: str
    package_path: Path


@dataclass(frozen=True, slots=True)
class PackageDescriptor:
    """
    只读取 package.yaml 得到的轻量信息。

    此阶段不加载 PyTorch 权重，用于：
    - 在真正加载模型前检查配置；
    - 计算所有模型共同的第一个决策终点；
    - 避免一次将全部模型同时放入显存。
    """
    candidate_id: str
    package_path: Path
    package_yaml_sha256: str

    package_id: str
    model_type: str
    model_name: str

    class_names: tuple[str, ...]
    window_sec: float
    step_sec: float


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    config_path: Path

    device: str
    verify_hashes: bool

    data_path: Path
    session: str

    step_sec: float
    warmup_windows: int
    measured_windows: int
    align_decision_endpoints: bool

    output_root: Path
    run_id: str

    candidates: tuple[CandidateSpec, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def required_mapping(
    payload: dict[str, Any],
    key: str,
    *,
    source: Path,
) -> dict[str, Any]:
    value = payload.get(key)

    if not isinstance(value, dict):
        raise ValueError(
            f"{source}: {key!r} must be a mapping."
        )

    return dict(value)


def required_string(
    payload: dict[str, Any],
    key: str,
    *,
    source: Path,
) -> str:
    value = payload.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{source}: {key!r} must be a non-empty string."
        )

    return value.strip()


def parse_candidates(
    *,
    benchmark_config: dict[str, Any],
    source: Path,
) -> tuple[CandidateSpec, ...]:
    raw_candidates = benchmark_config.get("candidates")

    if not isinstance(raw_candidates, list):
        raise ValueError(
            f"{source}: benchmark.candidates must be a list."
        )

    if not raw_candidates:
        raise ValueError(
            f"{source}: benchmark.candidates cannot be empty."
        )

    candidates: list[CandidateSpec] = []
    seen_ids: set[str] = set()

    for index, raw_candidate in enumerate(
        raw_candidates
    ):
        if not isinstance(raw_candidate, dict):
            raise ValueError(
                f"{source}: candidate index {index} "
                "must be a mapping."
            )

        candidate_id = required_string(
            raw_candidate,
            "id",
            source=source,
        )

        if candidate_id in seen_ids:
            raise ValueError(
                f"{source}: duplicated candidate id "
                f"{candidate_id!r}."
            )

        package_value = required_string(
            raw_candidate,
            "package",
            source=source,
        )

        package_path = resolve_path(package_value)

        if not package_path.is_dir():
            raise FileNotFoundError(
                f"{source}: package directory for "
                f"{candidate_id!r} was not found: "
                f"{package_path}"
            )

        candidates.append(
            CandidateSpec(
                candidate_id=candidate_id,
                package_path=package_path,
            )
        )
        seen_ids.add(candidate_id)

    return tuple(candidates)


def resolve_settings(
    *,
    config_path: Path,
    payload: dict[str, Any],
    device_override: str | None,
    output_root_override: str | None,
    run_id_override: str | None,
    no_verify_hashes: bool,
) -> BenchmarkSettings:
    benchmark = required_mapping(
        payload,
        "benchmark",
        source=config_path,
    )

    data = required_mapping(
        benchmark,
        "data",
        source=config_path,
    )

    schedule = required_mapping(
        benchmark,
        "schedule",
        source=config_path,
    )

    output = required_mapping(
        benchmark,
        "output",
        source=config_path,
    )

    device = (
        str(device_override)
        if device_override is not None
        else str(benchmark.get("device", "cpu"))
    )

    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError(
            "benchmark.device must be cpu, cuda, or mps; "
            f"got {device!r}."
        )

    data_path = resolve_path(
        required_string(
            data,
            "path",
            source=config_path,
        )
    )

    if not data_path.is_file():
        raise FileNotFoundError(
            f"HDF5 replay data was not found: {data_path}"
        )

    session = required_string(
        data,
        "session",
        source=config_path,
    )

    step_sec = float(schedule.get("step_sec", 0.5))
    warmup_windows = int(
        schedule.get("warmup_windows", 20)
    )
    measured_windows = int(
        schedule.get("measured_windows", 200)
    )

    if step_sec <= 0:
        raise ValueError(
            "benchmark.schedule.step_sec must be positive."
        )

    if warmup_windows < 0:
        raise ValueError(
            "benchmark.schedule.warmup_windows "
            "must be >= 0."
        )

    if measured_windows <= 0:
        raise ValueError(
            "benchmark.schedule.measured_windows "
            "must be positive."
        )

    default_run_id = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    output_root = (
        resolve_path(output_root_override)
        if output_root_override is not None
        else resolve_path(
            required_string(
                output,
                "root_dir",
                source=config_path,
            )
        )
    )

    run_id = (
        str(run_id_override)
        if run_id_override is not None
        else default_run_id
    )

    if not run_id.strip():
        raise ValueError("--run-id cannot be empty.")

    return BenchmarkSettings(
        config_path=config_path,
        device=device,
        verify_hashes=not no_verify_hashes,

        data_path=data_path,
        session=session,

        step_sec=step_sec,
        warmup_windows=warmup_windows,
        measured_windows=measured_windows,
        align_decision_endpoints=bool(
            schedule.get(
                "align_decision_endpoints",
                True,
            )
        ),

        output_root=output_root,
        run_id=run_id,

        candidates=parse_candidates(
            benchmark_config=benchmark,
            source=config_path,
        ),
    )


def read_package_descriptor(
    candidate: CandidateSpec,
) -> PackageDescriptor:
    package_yaml = candidate.package_path / "package.yaml"

    if not package_yaml.is_file():
        raise FileNotFoundError(
            f"package.yaml was not found: {package_yaml}"
        )

    payload = load_yaml(package_yaml)

    if int(payload.get("schema_version", -1)) != 2:
        raise ValueError(
            f"{package_yaml}: expected schema_version=2."
        )

    package = required_mapping(
        payload,
        "package",
        source=package_yaml,
    )

    model = required_mapping(
        payload,
        "model",
        source=package_yaml,
    )

    input_contract = required_mapping(
        payload,
        "input_contract",
        source=package_yaml,
    )

    runtime = required_mapping(
        payload,
        "runtime",
        source=package_yaml,
    )

    class_names = model.get("class_names")

    if (
        not isinstance(class_names, list)
        or not class_names
    ):
        raise ValueError(
            f"{package_yaml}: model.class_names "
            "must be a non-empty list."
        )

    window_sec = float(
        input_contract["window_sec"]
    )
    step_sec = float(runtime["step_sec"])

    if window_sec <= 0:
        raise ValueError(
            f"{package_yaml}: window_sec must be positive."
        )

    if step_sec <= 0:
        raise ValueError(
            f"{package_yaml}: step_sec must be positive."
        )

    if step_sec > window_sec:
        raise ValueError(
            f"{package_yaml}: step_sec must not exceed "
            "window_sec."
        )

    return PackageDescriptor(
        candidate_id=candidate.candidate_id,
        package_path=candidate.package_path,
        package_yaml_sha256=sha256_file(
            package_yaml
        ),

        package_id=required_string(
            package,
            "id",
            source=package_yaml,
        ),
        model_type=required_string(
            model,
            "type",
            source=package_yaml,
        ),
        model_name=required_string(
            model,
            "name",
            source=package_yaml,
        ),

        class_names=tuple(
            str(name)
            for name in class_names
        ),
        window_sec=window_sec,
        step_sec=step_sec,
    )


def validate_descriptors(
    *,
    descriptors: tuple[PackageDescriptor, ...],
    settings: BenchmarkSettings,
    dataset_class_names: tuple[str, ...],
) -> None:
    if not descriptors:
        raise ValueError("No package descriptors found.")

    for descriptor in descriptors:
        if not np.isclose(
            descriptor.step_sec,
            settings.step_sec,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                "All packages must use the benchmark "
                "step_sec. "
                f"candidate={descriptor.candidate_id}, "
                f"package_step={descriptor.step_sec}, "
                f"benchmark_step={settings.step_sec}."
            )

        if descriptor.class_names != dataset_class_names:
            raise ValueError(
                "Dataset class order does not match package "
                f"{descriptor.candidate_id!r}: "
                f"dataset={dataset_class_names}, "
                f"package={descriptor.class_names}."
            )


def validate_loaded_package(
    *,
    descriptor: PackageDescriptor,
    loaded: LoadedRuntimePackage,
    settings: BenchmarkSettings,
) -> None:
    if loaded.is_test_head:
        raise ValueError(
            f"Candidate {descriptor.candidate_id!r} is "
            "marked is_test_head=true. Formal benchmarks "
            "must use trained heads."
        )

    if loaded.model_type != descriptor.model_type:
        raise RuntimeError(
            "Loaded package model_type differs from its "
            f"package.yaml: {loaded.model_type!r} != "
            f"{descriptor.model_type!r}."
        )

    if not np.isclose(
        loaded.window_sec,
        descriptor.window_sec,
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Loaded package window_sec differs from its "
            f"package.yaml: {loaded.window_sec} != "
            f"{descriptor.window_sec}."
        )

    if not np.isclose(
        loaded.step_sec,
        settings.step_sec,
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Loaded package step_sec differs from the "
            f"benchmark step_sec: {loaded.step_sec} != "
            f"{settings.step_sec}."
        )


def release_model_memory(device: str) -> None:
    """
    每个 candidate 测完后释放 Python 引用；
    CUDA 上同时清缓存，避免下一个 package 被前一个占用显存。
    """
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a model-agnostic replay latency benchmark "
            "over Runtime Model Packages."
        )
    )

    parser.add_argument(
        "--config",
        default=(
            "configs/benchmarks/"
            "window_latency.yaml"
        ),
    )

    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default=None,
        help=(
            "Optional override for benchmark.device in YAML."
        ),
    )

    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Optional override for benchmark.output.root_dir. "
            "A timestamped run directory is still created."
        ),
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional reproducible output subdirectory name. "
            "Default: UTC timestamp."
        ),
    )

    parser.add_argument(
        "--no-verify-hashes",
        action="store_true",
        help=(
            "Skip package-file SHA-256 verification. "
            "Use only for local debugging."
        ),
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    config_path = resolve_path(args.config)
    config_payload = load_yaml(config_path)

    settings = resolve_settings(
        config_path=config_path,
        payload=config_payload,
        device_override=args.device,
        output_root_override=args.output_root,
        run_id_override=args.run_id,
        no_verify_hashes=args.no_verify_hashes,
    )

    output_dir = (
        settings.output_root / settings.run_id
    )

    if output_dir.exists():
        raise FileExistsError(
            f"Benchmark output directory already exists: "
            f"{output_dir}. Choose another --run-id."
        )

    dataset = EEGHDF5(settings.data_path)
    dataset_metadata = dataset.metadata

    dataset_class_names = tuple(
        str(name)
        for name in dataset_metadata.class_names
    )

    # 此处只读 package.yaml，不加载模型权重。
    descriptors = tuple(
        read_package_descriptor(candidate)
        for candidate in settings.candidates
    )

    validate_descriptors(
        descriptors=descriptors,
        settings=settings,
        dataset_class_names=dataset_class_names,
    )

    # 所有模型的第一个输出都对齐到“最大窗口长度”的结束时刻。
    maximum_window_sec = max(
        descriptor.window_sec
        for descriptor in descriptors
    )

    common_first_end_sample = (
        int(
            round(
                maximum_window_sec
                * float(dataset_metadata.sample_rate)
            )
        )
        if settings.align_decision_endpoints
        else None
    )

    total_provider_windows = (
        settings.warmup_windows
        + settings.measured_windows
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),

        "mode": "replay",
        "device": settings.device,
        "verify_hashes": settings.verify_hashes,

        "data_path": str(settings.data_path),
        "session": settings.session,
        "source_sample_rate": float(
            dataset_metadata.sample_rate
        ),
        "source_unit": str(dataset_metadata.unit),
        "source_channel_names": [
            str(name)
            for name in dataset_metadata.channel_names
        ],
        "class_names": list(dataset_class_names),

        "shared_step_sec": settings.step_sec,
        "warmup_windows": settings.warmup_windows,
        "measured_windows": settings.measured_windows,
        "align_decision_endpoints": (
            settings.align_decision_endpoints
        ),
        "common_first_end_sample": (
            common_first_end_sample
        ),

        "latency_definition": {
            "preprocessing_ms": (
                "RuntimeModel.prepare(raw_window), with "
                "device synchronization after completion."
            ),
            "inference_ms": (
                "RuntimeModel.predict_prepared(prepared), with "
                "device synchronization after completion."
            ),
            "output_materialization_ms": (
                "Move probabilities to CPU for a usable "
                "prediction result."
            ),
            "compute_total_ms": (
                "prepare + predict_prepared + probability "
                "materialization; excludes replay window "
                "generation, model loading, logging, and CSV I/O."
            ),
        },

        "candidates": [
            {
                "candidate_id": descriptor.candidate_id,
                "package_path": str(
                    descriptor.package_path
                ),
                "package_yaml_sha256": (
                    descriptor.package_yaml_sha256
                ),
                "package_id": descriptor.package_id,
                "model_type": descriptor.model_type,
                "model_name": descriptor.model_name,
                "window_sec": descriptor.window_sec,
                "step_sec": descriptor.step_sec,
            }
            for descriptor in descriptors
        ],
    }

    dump_json(
        manifest,
        output_dir / "benchmark_manifest.json",
    )

    all_rows: list[dict[str, object]] = []
    all_summaries = []

    print("=" * 78)
    print("Replay window latency benchmark")
    print("=" * 78)
    print("data:", settings.data_path)
    print("session:", settings.session)
    print("device:", settings.device)
    print("step_sec:", settings.step_sec)
    print(
        "warmup / measured windows:",
        settings.warmup_windows,
        "/",
        settings.measured_windows,
    )
    print(
        "common first decision end sample:",
        common_first_end_sample,
    )
    print("output:", output_dir)
    print()

    for position, descriptor in enumerate(
        descriptors,
        start=1,
    ):
        print(
            f"[{position}/{len(descriptors)}] "
            f"loading {descriptor.candidate_id} "
            f"({descriptor.model_type}, "
            f"{descriptor.window_sec:g}s)"
        )

        loaded: LoadedRuntimePackage | None = None
        core: RuntimeBenchmarkCore | None = None

        try:
            loaded = load_runtime_package(
                descriptor.package_path,
                device=settings.device,
                verify_hashes=settings.verify_hashes,
            )

            validate_loaded_package(
                descriptor=descriptor,
                loaded=loaded,
                settings=settings,
            )

            provider = ReplayWindowProvider(
                data_path=settings.data_path,
                session=settings.session,
                window_sec=loaded.window_sec,
                step_sec=settings.step_sec,
                maximum_windows=total_provider_windows,
                first_end_sample=(
                    common_first_end_sample
                ),
            )

            core = RuntimeBenchmarkCore(
                runtime_model=loaded.runtime_model,
                device=settings.device,
            )

            records = core.run(
                provider=provider,
                warmup_windows=settings.warmup_windows,
                measured_windows=settings.measured_windows,
            )

            candidate = BenchmarkCandidate(
                candidate_id=descriptor.candidate_id,
                model_name=loaded.model_name,
                model_type=loaded.model_type,
                package_path=str(
                    descriptor.package_path
                ),

                # 这里记录 package.yaml 的 SHA-256，
                # 完整权重哈希仍保留在 package.yaml 内。
                package_sha256=(
                    descriptor.package_yaml_sha256
                ),

                window_sec=float(loaded.window_sec),
                step_sec=float(loaded.step_sec),

                device=settings.device,
                source_mode="replay",

                warmup_windows=settings.warmup_windows,
                measured_windows=settings.measured_windows,
            )

            all_rows.extend(
                flatten_window_record(
                    candidate=candidate,
                    record=record,
                )
                for record in records
            )

            summary = build_candidate_summary(
                candidate=candidate,
                records=records,
            )

            all_summaries.append(summary)

            print(
                f"  compute_total_ms: "
                f"P50={summary.compute_total_ms['p50']:.3f} ms, "
                f"P95={summary.compute_total_ms['p95']:.3f} ms"
            )

        finally:
            # 不让前一个模型长期占用 GPU / MPS 内存。
            del core
            del loaded
            release_model_memory(settings.device)

    records_path = write_window_records_csv(
        path=output_dir / "window_records.csv",
        rows=all_rows,
    )

    summary_path = write_benchmark_summary_json(
        path=output_dir / "summary.json",
        summaries=all_summaries,
        benchmark_metadata={
            "manifest_path": str(
                output_dir / "benchmark_manifest.json"
            ),
            "mode": "replay",
            "device": settings.device,
            "source_data": str(settings.data_path),
            "session": settings.session,
            "shared_step_sec": settings.step_sec,
            "warmup_windows": settings.warmup_windows,
            "measured_windows": settings.measured_windows,
            "common_first_end_sample": (
                common_first_end_sample
            ),
        },
    )

    print()
    print("=" * 78)
    print("Benchmark completed")
    print("=" * 78)
    print("manifest:", output_dir / "benchmark_manifest.json")
    print("window records:", records_path)
    print("summary:", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())