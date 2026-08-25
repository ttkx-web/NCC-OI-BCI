from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bci_dayloop.training.model_50m import artifacts
from bci_dayloop.training.model_50m import runner
from bci_dayloop.training.model_50m.linear_head import WindowSet


def test_artifact_path_preparation_preserves_output_before_run_dir_guard(tmp_path) -> None:
    """Keep the historical directory-creation order behind --overwrite."""
    checkpoint = tmp_path / "backbone.pt"
    checkpoint.write_bytes(b"backbone")
    run_dir = tmp_path / "existing-run"
    run_dir.mkdir()
    output_path = tmp_path / "new-parent" / "head.pt"
    with pytest.raises(FileExistsError):
        artifacts.prepare_training_artifact_paths(
            run_dir=run_dir,
            output_path=output_path,
            overwrite=False,
            backbone_checkpoint=checkpoint,
        )
    assert output_path.parent.is_dir()


@pytest.mark.parametrize(
    ("partial_enabled", "lora_enabled", "expected_format"),
    [(False, False, 2), (True, False, 3), (False, True, 4)],
)
def test_population_artifact_writer_preserves_mode_specific_checkpoint_contract(
    monkeypatch,
    tmp_path,
    partial_enabled,
    lora_enabled,
    expected_format,
) -> None:
    """The runner delegates the established files and optional states intact."""
    checkpoint_calls: list[dict[str, object]] = []

    def fake_save_classifier_checkpoint(**kwargs):
        checkpoint_calls.append(kwargs)
        path = kwargs["checkpoint_path"]
        torch.save({"format_version": expected_format}, path)
        return path

    monkeypatch.setattr(artifacts, "save_classifier_checkpoint", fake_save_classifier_checkpoint)

    class FakeClassifier:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.head = torch.nn.Linear(2, 2)
            self.backbone = SimpleNamespace(model=torch.nn.Linear(2, 2))

    window_set = WindowSet(
        windows=np.zeros((2, 1, 2), dtype=np.float32),
        labels=np.asarray([0, 1], dtype=np.int64),
        source_trial_ids=((1,), (2,)),
        construction="direct_source_trial",
    )
    split = SimpleNamespace(
        bundle=SimpleNamespace(window_set=window_set),
        source_trial_summary={"count": 2},
    )
    metric = SimpleNamespace(
        to_dict=lambda: {"accuracy": 0.5, "loss": 1.0},
    )
    classifier = FakeClassifier()
    args = SimpleNamespace(
        split_mode="loso",
        data_reader="eeg",
        validation_session="1test",
        lora_target_modules=("q",),
        window_construction="direct_trial",
        embedding_layer="output",
        unfreeze_last_n_blocks=1,
        lora_last_n_blocks=1,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        head_lr=1e-3,
        backbone_lr=1e-4,
        lora_lr=5e-4,
        momentum=0.9,
        weight_decay=0.01,
        seed=7,
        window_seed=9,
        metric_for_best="val_bacc",
        window_sec=4.0,
        window_stride_sec=4.0,
        feature_cache_dtype="float16",
        epochs=2,
        patience=1,
        validation_ratio=0.2,
    )
    config = SimpleNamespace(
        output_layer_idx=8,
        model_n_time_patches=10,
        target_sample_rate=100.0,
        target_num_points=400,
        num_tokens=10,
        patch_num_points=100,
        aggregation="flatten",
        classifier_input_dim=2,
        head_type="linear",
        head_hidden_dim=8,
        head_dropout=0.0,
        head_norm="none",
    )
    backbone_checkpoint = tmp_path / "backbone.pt"
    backbone_checkpoint.write_bytes(b"backbone")
    prepared = artifacts.prepare_training_artifact_paths(
        run_dir=tmp_path / "run",
        output_path=tmp_path / "heads" / "head.pt",
        overwrite=False,
        backbone_checkpoint=backbone_checkpoint,
    )
    artifacts.save_initial_run_config(
        paths=prepared,
        timestamp="20260825_000000",
        target_subject=1,
        population_subjects=(2,),
        split_mode="loso",
        loso_train_session="0train",
        within_subject_train_sessions=(),
        validation_session="1test",
        final_test_session="1test",
        backbone_checkpoint=backbone_checkpoint,
        validation_ratio=0.2,
        data_root=tmp_path,
        data_pattern="subject_{subject:02d}.h5",
        data_reader="eeg",
        subject_identities={"1": {"subject_id": 1}},
        subject_paths={1: tmp_path / "subject_01.h5"},
        arguments=vars(args),
    )
    result = artifacts.save_training_artifacts(
        artifacts.TrainingArtifactInputs(
            run_dir=prepared.run_dir,
            output_path=prepared.output_path,
            classifier=classifier,
            args=args,
            config=config,
            backbone=SimpleNamespace(trainable_parameters=3),
            checkpoint_path=backbone_checkpoint,
            git_commit="abc",
            backbone_sha256=prepared.backbone_sha256,
            target_subject=1,
            population_subjects=(2,),
            subjects=(1, 2),
            loso_train_session="0train",
            within_subject_train_sessions=(),
            final_test_session="1test",
            embedding_layer=8,
            subject_identities={"1": {"subject_id": 1}},
            all_subject_paths={1: tmp_path / "subject_01.h5"},
            metadata=SimpleNamespace(
                dataset_name="bnci2014_001", sample_rate=100.0,
                unit="uV", channel_names=("C1",),
            ),
            class_names=("left", "right"),
            label_mapping={0: "left", 1: "right"},
            num_classes=2,
            within_subject_metadata=None,
            backbone_adaptation=("lora" if lora_enabled else "partial" if partial_enabled else "frozen"),
            partial_finetuning_enabled=partial_enabled,
            lora_enabled=lora_enabled,
            trainable_backbone_parameters=list(classifier.backbone.model.parameters()) if partial_enabled else [],
            trainable_block_indices=(7,),
            lora_parameters=[],
            total_trainable_parameter_count=6,
            head_parameters=list(classifier.head.parameters()),
            selected_val_metrics=metric,
            target_metrics=metric,
            train_build=split,
            val_build=split,
            target_build=split,
            feature_cache_enabled=not (partial_enabled or lora_enabled),
            preprocessing_contract={"version": 1},
            preprocessing_hash="preprocess",
            model_load_seconds=0.1,
            training_seconds=0.2,
            best_epoch=1,
            epoch_rows=[{"epoch": 1, "train_loss": 1.0}],
        )
    )

    assert runner.atomic_write_json is artifacts.atomic_write_json
    assert (checkpoint_calls[0]["backbone_state_dict"] is not None) is partial_enabled
    assert (checkpoint_calls[0]["lora_state_dict"] is not None) is lora_enabled
    assert result.checkpoint_path.exists()
    assert result.epoch_metrics_path.read_text(encoding="utf-8") == "epoch,train_loss\n1,1.0\n"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert report["training"]["best_epoch"] == 1
    assert summary["classifier_checkpoint"] == str(result.checkpoint_path)
    assert json.loads((prepared.run_dir / "run_config.json").read_text())["backbone_checkpoint"] == str(backbone_checkpoint)
