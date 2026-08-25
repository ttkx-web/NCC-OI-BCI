"""Aggregate a complete subject/model matrix of NeuroOnline evaluations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

if __package__:
    from scripts.report_seed_loso_results import (
        METRICS,
        _subject_result,
        build_summary,
        write_reports,
    )
else:
    from report_seed_loso_results import (
        METRICS,
        _subject_result,
        build_summary,
        write_reports,
    )


DEFAULT_SUMMARY_TEMPLATE = (
    "subject_{subject:02d}/{model}/evaluation/summary.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--subjects", required=True, nargs="+", type=int)
    parser.add_argument("--models", required=True, nargs="+")
    parser.add_argument(
        "--summary-template",
        default=DEFAULT_SUMMARY_TEMPLATE,
        help=(
            "Package-independent relative path template with {subject} and "
            "{model} fields."
        ),
    )
    parser.add_argument(
        "--primary-metric",
        choices=METRICS,
        default="balanced_accuracy",
    )
    parser.add_argument(
        "--filename-prefix",
        default="neuroonline_benchmark",
    )
    parser.add_argument(
        "--markdown-filename",
        help=(
            "Markdown filename to write inside output-dir. Use this when a "
            "consumer expects a fixed report name."
        ),
    )
    return parser


def load_benchmark_results(
    *,
    input_root: Path,
    subjects: Sequence[int],
    models: Sequence[str],
    summary_template: str = DEFAULT_SUMMARY_TEMPLATE,
) -> list[dict[str, object]]:
    if not input_root.is_dir():
        raise ValueError(f"Benchmark input root does not exist: {input_root}")
    normalized_subjects = tuple(int(subject) for subject in subjects)
    normalized_models = tuple(str(model).strip() for model in models)
    if not normalized_subjects or len(set(normalized_subjects)) != len(
        normalized_subjects
    ):
        raise ValueError("subjects must be non-empty and unique.")
    if (
        not normalized_models
        or any(not model for model in normalized_models)
        or len(set(normalized_models)) != len(normalized_models)
    ):
        raise ValueError("models must be non-empty and unique.")

    root = input_root.resolve()
    results: list[dict[str, object]] = []
    for subject in normalized_subjects:
        for model in normalized_models:
            try:
                relative = Path(
                    summary_template.format(subject=subject, model=model)
                )
            except (KeyError, ValueError) as error:
                raise ValueError(
                    "summary_template must contain valid {subject} and "
                    "{model} fields."
                ) from error
            if relative.is_absolute():
                raise ValueError("summary_template must produce relative paths.")
            summary_path = (root / relative).resolve()
            try:
                summary_path.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    "summary_template must not escape input_root."
                ) from error
            results.append(
                _subject_result(
                    subject=subject,
                    model=model,
                    path=summary_path,
                )
            )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        results = load_benchmark_results(
            input_root=args.input_root,
            subjects=args.subjects,
            models=args.models,
            summary_template=args.summary_template,
        )
        report = build_summary(
            results,
            dataset_name=args.dataset_name,
            subjects=args.subjects,
            models=args.models,
            primary_metric=args.primary_metric,
        )
        outputs = write_reports(
            results=results,
            report=report,
            output_dir=args.output_dir,
            filename_prefix=args.filename_prefix,
            markdown_filename=args.markdown_filename,
        )
    except ValueError as error:
        parser.error(str(error))
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
