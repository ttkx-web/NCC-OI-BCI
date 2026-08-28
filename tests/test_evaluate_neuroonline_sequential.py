from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from bci_dayloop.data.hdf5_dataset import HDF5Metadata, write_hdf5
from bci_dayloop.runtime.adaptation_types import OnlineUpdateResult
from bci_dayloop.runtime.model import RuntimeModel
from bci_dayloop.runtime.types import (
    CanonicalEEGWindow,
    InputContract,
    ModelOutput,
    ModelTensor,
    PreparedModelInput,
)
from bci_dayloop.preprocessing.base import ModelInputTransform
from bci_dayloop.preprocessing.canonical import SignalCanonicalizer
from bci_dayloop.models.base import ModelBackend

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts import evaluate_neuroonline_sequential as seq


CLASS_NAMES = ("left_hand", "right_hand", "feet", "tongue")


class SequenceInputTransform(ModelInputTransform):
    def __init__(self, *, sample_rate: float = 2.0, window_sec: float = 4.0) -> None:
        self._contract = InputContract(
            channel_names=("C3", "C4"),
            sample_rate=sample_rate,
            window_sec=window_sec,
            num_samples=int(round(sample_rate * window_sec)),
            input_unit="uV",
            tensor_layout="BCT",
            strict_window_duration=True,
            model_input_keys=("signal",),
        )

    @property
    def input_contract(self) -> InputContract:
        return self._contract

    def transform(self, window: CanonicalEEGWindow) -> PreparedModelInput:
        signal = torch.from_numpy(
            np.ascontiguousarray(window.data, dtype=np.float32)
        ).unsqueeze(0)
        return PreparedModelInput(
            model_input={"signal": signal},
            canonical_window=window,
            preprocessing_trace=["sequence_input_transform"],
            diagnostics={"raw_label": window.label},
        )


class SequenceBackend(ModelBackend):
    def __init__(self, predictions: list[int]) -> None:
        self.predictions = list(predictions)
        self.calls = 0

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def num_classes(self) -> int:
        return len(CLASS_NAMES)

    def predict_tensor(
        self,
        model_input: ModelTensor,
        return_features: bool = False,
    ) -> ModelOutput:
        del model_input, return_features
        if self.calls >= len(self.predictions):
            raise RuntimeError("No fake prediction left.")
        predicted = int(self.predictions[self.calls])
        self.calls += 1
        return output_for(predicted)

    def encode_tensor(self, model_input: ModelTensor) -> torch.Tensor:
        if not isinstance(model_input, dict):
            raise TypeError("Expected dict model input.")
        return model_input["signal"].mean(dim=-1)

    def get_trainable_parameters(self, scope: str):
        del scope
        return []


class RecordingRuntimeModel(RuntimeModel):
    def __init__(self, predictions: list[int]) -> None:
        super().__init__(
            canonicalizer=SignalCanonicalizer(target_unit="uV"),
            input_transform=SequenceInputTransform(),
            backend=SequenceBackend(predictions),
        )
        self.prepare_labels: list[int | None] = []

    def prepare(self, raw_window):
        self.prepare_labels.append(raw_window.label)
        return super().prepare(raw_window)


def output_for(predicted: int) -> ModelOutput:
    probabilities = torch.full((1, len(CLASS_NAMES)), 0.05, dtype=torch.float32)
    probabilities[0, predicted] = 0.85
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return ModelOutput(
        logits=torch.log(probabilities.clamp_min(1e-8)),
        probabilities=probabilities,
        predicted_class=predicted,
        confidence=float(probabilities[0, predicted].item()),
        diagnostics={"predicted": predicted},
    )


@dataclass
class FakeLoadedPackage:
    runtime_model: RuntimeModel
    package_path: Path
    model_type: str = "fake"
    model_name: str = "fake-runtime"
    class_names: tuple[str, ...] = CLASS_NAMES
    command_map: dict[str, str] = None  # type: ignore[assignment]
    step_sec: float = 0.5
    confidence_threshold: float = 0.55
    is_test_head: bool = False
    warning_message: str | None = None
    metrics: dict = None  # type: ignore[assignment]
    package_metadata: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.command_map is None:
            self.command_map = {}
        if self.metrics is None:
            self.metrics = {}
        if self.package_metadata is None:
            self.package_metadata = {}

    @property
    def window_sec(self) -> float:
        return float(self.runtime_model.input_contract.window_sec)

    @property
    def target_sample_rate(self) -> float:
        return float(self.runtime_model.input_contract.sample_rate)


class FakeNeuroOnlineStrategy:
    events: list[str] = []
    used_revisions: list[str] = []
    instances: list["FakeNeuroOnlineStrategy"] = []

    def __init__(self, config) -> None:
        self.config = config
        self.update_step = 0
        self.model_revision = "neuroonline-0"
        self._prediction_index = 0
        self._last_observation_id: str | None = None
        self.predictions = [0, 1, 2, 3]
        FakeNeuroOnlineStrategy.instances.append(self)

    def initialize(self, *, runtime_model, context) -> None:
        del runtime_model, context
        FakeNeuroOnlineStrategy.events.append("initialize")

    def predict_prepared(self, prepared, *, return_features: bool = False):
        del prepared, return_features
        FakeNeuroOnlineStrategy.events.append(f"predict:{self.update_step}")
        FakeNeuroOnlineStrategy.used_revisions.append(self.model_revision)
        predicted = self.predictions[self._prediction_index]
        self._prediction_index += 1
        return output_for(predicted)

    def observe(self, observation) -> None:
        FakeNeuroOnlineStrategy.events.append("observe")
        self._last_observation_id = observation.observation_id

    def submit_feedback(self, feedback) -> None:
        FakeNeuroOnlineStrategy.events.append(f"feedback:{feedback.label}")
        assert feedback.observation_id == self._last_observation_id

    def maybe_update(self, *, runtime_model):
        del runtime_model
        FakeNeuroOnlineStrategy.events.append("update")
        if self._prediction_index == 2:
            self.update_step = 1
            self.model_revision = "neuroonline-1"
            return OnlineUpdateResult(
                strategy_name="neuroonline",
                applied=True,
                update_step=1,
                model_revision="neuroonline-1",
                samples_used=2,
                latency_ms=1.5,
                metrics={"loss": 0.25},
            )
        return OnlineUpdateResult(
            strategy_name="neuroonline",
            applied=False,
            update_step=self.update_step,
            model_revision=self.model_revision,
            reason="waiting",
        )


def metadata(sample_rate: float = 2.0) -> HDF5Metadata:
    return HDF5Metadata(
        sample_rate=sample_rate,
        channel_names=["C3", "C4"],
        class_names=list(CLASS_NAMES),
        unit="uV",
        dataset_name="fake",
    )


def write_dataset(
    path: Path,
    *,
    labels: list[int] | None = None,
    sample_rate: float = 2.0,
    samples: int | None = None,
    trial_ids: list[int] | None = None,
) -> Path:
    labels = labels or [0, 1, 2, 3]
    samples = samples if samples is not None else int(round(sample_rate * 4.0))
    data = np.arange(len(labels) * 2 * samples, dtype=np.float32).reshape(
        len(labels), 2, samples
    )
    subject_ids = np.ones(len(labels), dtype=np.int64)
    trial_values = np.asarray(
        trial_ids if trial_ids is not None else list(range(10, 10 + len(labels))),
        dtype=np.int64,
    )
    return write_hdf5(
        path,
        data,
        np.asarray(labels, dtype=np.int64),
        subject_ids,
        ["1test"] * len(labels),
        trial_values,
        metadata(sample_rate),
    )


def settings(tmp_path: Path, *, data_path: Path | None = None, max_trials=None):
    args = seq.build_parser().parse_args(
        [
            "--config",
            "configs/stage0/day1_bnci_s01.yaml",
            "--data",
            str(data_path or tmp_path / "data.h5"),
            "--model-package",
            str(tmp_path / "pkg"),
            "--output-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--online-strategy",
            "both",
            *([] if max_trials is None else ["--max-trials", str(max_trials)]),
        ]
    )
    return seq.resolve_settings(
        args,
        {
            "project": {"run_dir": str(tmp_path / "runs")},
            "online": {
                "strategy": "none",
                "neuroonline": {
                    "warmup_feedback": 2,
                    "update_interval": 2,
                    "recent_buffer_size": 4,
                },
            },
        },
    )


def test_metrics_accuracy_balanced_accuracy_and_macro_f1():
    metrics = seq._classification_metrics(
        [0, 1, 2, 3],
        [0, 1, 1, 3],
        class_names=CLASS_NAMES,
    )
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx(
        (1.0 + (2 / 3) + 0.0 + 1.0) / 4
    )


def test_cli_online_strategy_overrides_yaml(tmp_path):
    args = seq.build_parser().parse_args(
        [
            "--config",
            "configs/stage0/day1_bnci_s01.yaml",
            "--data",
            str(tmp_path / "data.h5"),
            "--model-package",
            str(tmp_path / "pkg"),
            "--online-strategy",
            "both",
        ]
    )
    resolved = seq.resolve_settings(
        args,
        {
            "project": {"run_dir": str(tmp_path / "runs")},
            "online": {"strategy": "none"},
        },
    )
    assert resolved.online_strategy == "both"


def test_post_warmup_starts_after_warmup_feedback():
    records = [
        {
            "trial_ordinal": index + 1,
            "true_class_id": true,
            "predicted_class_id": pred,
            "update_step_used": 0,
        }
        for index, (true, pred) in enumerate(
            [(0, 0), (1, 0), (2, 2), (3, 3)]
        )
    ]
    metrics = seq.compute_metrics_for_records(
        records,
        class_names=CLASS_NAMES,
        warmup_feedback=2,
        block_size=2,
    )
    assert metrics["warmup_predictions"]["num_samples"] == 2
    assert metrics["post_warmup"]["num_samples"] == 2
    assert metrics["post_warmup"]["accuracy"] == pytest.approx(1.0)


def test_validates_exact_four_second_trials(tmp_path):
    data_path = write_dataset(tmp_path / "data.h5")
    loaded = seq.load_sequential_dataset(settings(tmp_path, data_path=data_path))
    assert loaded.data.shape == (4, 2, 8)


def test_dataset_package_window_mismatch_fails_closed(tmp_path):
    data_path = write_dataset(tmp_path / "bad.h5", samples=7)
    dataset = seq.load_sequential_dataset(settings(tmp_path, data_path=data_path))
    package = FakeLoadedPackage(
        RecordingRuntimeModel([0, 1, 2, 3]),
        tmp_path / "pkg",
    )
    with pytest.raises(ValueError, match="Dataset and Runtime Package window contracts differ"):
        seq.validate_package_contract(package, dataset=dataset)


def test_trial_order_is_not_shuffled(tmp_path):
    data_path = write_dataset(
        tmp_path / "data.h5",
        labels=[0, 1, 2],
        trial_ids=[30, 10, 20],
    )
    loaded = seq.load_sequential_dataset(settings(tmp_path, data_path=data_path))
    assert loaded.trial_ids.tolist() == [30, 10, 20]


def test_both_mode_loads_runtime_package_twice_and_keeps_instances_separate(tmp_path):
    data_path = write_dataset(tmp_path / "data.h5")
    cfg = settings(tmp_path, data_path=data_path)
    loaded_models: list[RecordingRuntimeModel] = []

    def loader(*_args, **_kwargs):
        runtime = RecordingRuntimeModel([0, 1, 2, 3])
        loaded_models.append(runtime)
        return FakeLoadedPackage(runtime, cfg.model_package)

    summary = seq.run_evaluation(
        cfg,
        package_loader=loader,
        strategy_factory=FakeNeuroOnlineStrategy,  # type: ignore[arg-type]
    )

    assert len(loaded_models) == 2
    assert loaded_models[0] is not loaded_models[1]
    assert summary["static"]["num_trials"] == 4
    assert summary["neuroonline"]["num_trials"] == 4
    assert summary["protocol"] == {
        "name": "neuroonline_sequential_trial_4s",
        "window_sec": 4.0,
        "step_sec": 4.0,
        "one_prediction_per_source_trial": True,
        "uses_replay_acquirer": False,
        "uses_sliding_window_decoder": False,
        "continuous_stream_concatenation": False,
        "shuffle": False,
        "label_available_during_prediction": False,
        "package_step_sec_ignored_for_trial_sequence": True,
    }


def test_predict_feedback_update_order_and_revisions(tmp_path):
    FakeNeuroOnlineStrategy.events = []
    FakeNeuroOnlineStrategy.used_revisions = []
    FakeNeuroOnlineStrategy.instances = []
    data_path = write_dataset(tmp_path / "data.h5")
    cfg = settings(tmp_path, data_path=data_path)

    def loader(*_args, **_kwargs):
        return FakeLoadedPackage(RecordingRuntimeModel([0, 1, 2, 3]), cfg.model_package)

    summary = seq.run_evaluation(
        cfg,
        package_loader=loader,
        strategy_factory=FakeNeuroOnlineStrategy,  # type: ignore[arg-type]
    )
    records = summary["neuroonline"]["updates"]["updates"]
    online_rows = [
        row
        for row in (cfg.output_dir / "trial_predictions.jsonl").read_text().splitlines()
        if '"mode":"neuroonline"' in row
    ]

    assert FakeNeuroOnlineStrategy.events[:5] == [
        "initialize",
        "predict:0",
        "observe",
        "feedback:0",
        "update",
    ]
    assert records[0]["trial_ordinal"] == 2
    assert FakeNeuroOnlineStrategy.used_revisions[:3] == [
        "neuroonline-0",
        "neuroonline-0",
        "neuroonline-1",
    ]
    assert '"model_revision_used":"neuroonline-0"' in online_rows[1]
    assert '"model_revision_after_prediction":"neuroonline-1"' in online_rows[1]
    assert '"model_revision_used":"neuroonline-1"' in online_rows[2]


def test_static_and_neuroonline_identity_before_first_update(tmp_path):
    data_path = write_dataset(tmp_path / "data.h5")
    cfg = settings(tmp_path, data_path=data_path)

    def loader(*_args, **_kwargs):
        return FakeLoadedPackage(RecordingRuntimeModel([0, 1, 2, 3]), cfg.model_package)

    summary = seq.run_evaluation(
        cfg,
        package_loader=loader,
        strategy_factory=FakeNeuroOnlineStrategy,  # type: ignore[arg-type]
    )
    check = summary["identity_initialization_check"]
    assert check["comparison_trial_count"] == 2
    assert check["prediction_agreement_rate"] == pytest.approx(1.0)
    assert check["maximum_probability_absolute_difference"] == pytest.approx(0.0)


def test_max_trials_warmup_warning(tmp_path):
    data_path = write_dataset(tmp_path / "data.h5")
    cfg = settings(tmp_path, data_path=data_path, max_trials=2)

    def loader(*_args, **_kwargs):
        return FakeLoadedPackage(RecordingRuntimeModel([0, 1]), cfg.model_package)

    with pytest.warns(RuntimeWarning, match="max_trials <= warmup_feedback"):
        summary = seq.run_evaluation(
            cfg,
            package_loader=loader,
            strategy_factory=FakeNeuroOnlineStrategy,  # type: ignore[arg-type]
        )
    assert summary["warnings"]


def test_static_mode_does_not_create_generator_or_call_online_interfaces(tmp_path):
    data_path = write_dataset(tmp_path / "data.h5")
    cfg = settings(tmp_path, data_path=data_path)
    cfg = dataclass_replace(cfg, online_strategy="none")
    FakeNeuroOnlineStrategy.instances = []

    def loader(*_args, **_kwargs):
        runtime = RecordingRuntimeModel([0, 1, 2, 3])
        return FakeLoadedPackage(runtime, Path(cfg.model_package))

    summary = seq.run_evaluation(
        cfg,  # type: ignore[arg-type]
        package_loader=loader,
        strategy_factory=FakeNeuroOnlineStrategy,  # type: ignore[arg-type]
    )
    assert not FakeNeuroOnlineStrategy.instances
    assert summary["static"]["num_trials"] == 4
    assert summary["neuroonline"] is None


def test_prepare_receives_label_none(tmp_path):
    data_path = write_dataset(tmp_path / "data.h5")
    cfg = settings(tmp_path, data_path=data_path)
    runtime = RecordingRuntimeModel([0, 1, 2, 3])

    def loader(*_args, **_kwargs):
        return FakeLoadedPackage(runtime, cfg.model_package)

    seq.run_evaluation(
        dataclass_replace(cfg, online_strategy="none"),
        package_loader=loader,
    )
    assert runtime.prepare_labels == [None, None, None, None]


def test_evaluation_order_defaults_to_persisted(tmp_path):
    args = seq.build_parser().parse_args(
        [
            "--data",
            str(tmp_path / "data.h5"),
            "--model-package",
            str(tmp_path / "pkg"),
        ]
    )
    assert args.evaluation_order == "persisted"
    assert args.order_seed == 20260826


def test_random_evaluation_indices_are_seeded_label_independent_bijection():
    first = seq.resolve_evaluation_indices(
        20, evaluation_order="random_permutation", order_seed=7
    )
    repeated = seq.resolve_evaluation_indices(
        20, evaluation_order="random_permutation", order_seed=7
    )
    different = seq.resolve_evaluation_indices(
        20, evaluation_order="random_permutation", order_seed=8
    )

    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, different)
    assert sorted(first.tolist()) == list(range(20))
    # The resolver accepts only N/order/seed, so labels cannot affect it.
    assert "labels" not in seq.resolve_evaluation_indices.__annotations__


def test_random_order_moves_trial_level_fields_together_and_remains_causal(tmp_path):
    FakeNeuroOnlineStrategy.events = []
    FakeNeuroOnlineStrategy.used_revisions = []
    FakeNeuroOnlineStrategy.instances = []
    data_path = write_dataset(
        tmp_path / "data.h5",
        labels=[0, 1, 2, 3],
        trial_ids=[30, 10, 20, 40],
    )
    cfg = dataclass_replace(
        settings(tmp_path, data_path=data_path),
        evaluation_order="random_permutation",
        order_seed=11,
    )

    class PairRecordingRuntimeModel(RecordingRuntimeModel):
        def __init__(self):
            super().__init__([0, 1, 2, 3])
            self.first_values: list[float] = []

        def prepare(self, raw_window):
            self.first_values.append(float(raw_window.data[0, 0]))
            return super().prepare(raw_window)

    runtimes: list[PairRecordingRuntimeModel] = []

    def loader(*_args, **_kwargs):
        runtime = PairRecordingRuntimeModel()
        runtimes.append(runtime)
        return FakeLoadedPackage(runtime, cfg.model_package)

    summary = seq.run_evaluation(
        cfg,
        package_loader=loader,
        strategy_factory=FakeNeuroOnlineStrategy,  # type: ignore[arg-type]
    )
    rows = [
        json.loads(line)
        for line in (cfg.output_dir / "trial_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if json.loads(line)["mode"] == "neuroonline"
    ]
    source_labels = [0, 1, 2, 3]
    source_ids = [30, 10, 20, 40]
    for evaluation_ordinal, row in enumerate(rows, start=1):
        source_index = row["source_dataset_index"]
        assert row["evaluation_ordinal"] == evaluation_ordinal
        assert row["trial_ordinal"] == evaluation_ordinal
        assert row["source_trial_ordinal"] == source_index + 1
        assert row["true_class_id"] == source_labels[source_index]
        assert int(row["source_trial_id"]) == source_ids[source_index]
        assert runtimes[1].first_values[evaluation_ordinal - 1] == pytest.approx(
            source_index * 2 * 8
        )

    assert FakeNeuroOnlineStrategy.events[:5] == [
        "initialize",
        "predict:0",
        "observe",
        f"feedback:{rows[0]['true_class_id']}",
        "update",
    ]
    assert summary["protocol"]["evaluation_order"] == "random_permutation"
    assert summary["protocol"]["shuffle"] is True
    assert summary["order_control"]["class_counts_before"] == summary[
        "order_control"
    ]["class_counts_after"]


def test_static_and_neuroonline_use_identical_random_permutation(tmp_path):
    data_path = write_dataset(tmp_path / "data.h5")
    cfg = dataclass_replace(
        settings(tmp_path, data_path=data_path),
        evaluation_order="random_permutation",
        order_seed=91,
    )

    def loader(*_args, **_kwargs):
        return FakeLoadedPackage(
            RecordingRuntimeModel([0, 1, 2, 3]), cfg.model_package
        )

    seq.run_evaluation(
        cfg,
        package_loader=loader,
        strategy_factory=FakeNeuroOnlineStrategy,  # type: ignore[arg-type]
    )
    rows = [
        json.loads(line)
        for line in (cfg.output_dir / "trial_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    by_mode = {
        mode: [row["source_dataset_index"] for row in rows if row["mode"] == mode]
        for mode in ("none", "neuroonline")
    }
    assert by_mode["none"] == by_mode["neuroonline"]
