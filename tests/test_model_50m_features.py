from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from bci_dayloop.training.model_50m import features
from bci_dayloop.training.model_50m import linear_head
from bci_dayloop.training.model_50m import runner
from bci_dayloop.training.model_50m.types import WindowBundle


def test_frozen_feature_extraction_and_population_cache_are_compatible(
    monkeypatch,
    tmp_path,
) -> None:
    """Exercise CPU extraction order and the existing write-only cache schema."""
    class FakePreprocessor:
        def __init__(self, config) -> None:
            del config

        def __call__(self, *, signal, **kwargs):
            del kwargs
            return SimpleNamespace(
                signal=np.asarray(signal, dtype=np.float32),
                mapped_channel_count=2,
                missing_channel_count=0,
            )

    class FakeTokenizer:
        def __init__(self, config) -> None:
            del config

        def __call__(self, processed):
            return torch.from_numpy(processed.signal.reshape(-1)[:3].copy())

    class FakeClassifier:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.extract_calls = 0

        def eval(self) -> None:
            return None

        def extract_features(self, batch):
            self.extract_calls += 1
            return batch + 10.0

    monkeypatch.setattr(features, "Model50MPreprocessor", FakePreprocessor)
    monkeypatch.setattr(features, "Model50MTokenizer", FakeTokenizer)
    monkeypatch.setattr(
        features,
        "stack_model50m_tokens",
        lambda samples, device: torch.stack(samples).to(device),
    )

    window_set = linear_head.WindowSet(
        windows=np.asarray(
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[4.0, 5.0], [6.0, 7.0]],
                [[8.0, 9.0], [10.0, 11.0]],
            ],
            dtype=np.float32,
        ),
        labels=np.asarray([1, 0, 1], dtype=np.int64),
        source_trial_ids=((10,), (11,), (12,)),
        construction="direct_source_trial",
    )
    classifier = FakeClassifier()
    dataset = features.extract_frozen_features(
        window_set=window_set,
        metadata=SimpleNamespace(
            channel_names=("C1", "C2"), sample_rate=100.0, unit="uV"
        ),
        config=SimpleNamespace(classifier_input_dim=3),
        classifier=classifier,
        preprocess_batch_size=2,
        cache_dtype=torch.float16,
        split_name="train",
        log_every=10,
    )
    extracted, labels = dataset.tensors
    assert classifier.extract_calls == 2
    assert extracted.dtype is torch.float16
    assert extracted.tolist() == [[10.0, 11.0, 12.0], [14.0, 15.0, 16.0], [18.0, 19.0, 20.0]]
    assert labels.tolist() == [1, 0, 1]

    bundle = WindowBundle(
        window_set=window_set,
        window_subject_ids=np.asarray([2, 2, 2], dtype=np.int64),
    )
    cache_path = tmp_path / "features.pt"
    assert features.population_feature_cache_path(
        tmp_path, split_name="population_train"
    ) == tmp_path / "features_population_train.pt"
    features.save_population_feature_cache(
        dataset=dataset,
        bundle=bundle,
        path=cache_path,
        split_name="population_train",
        class_names=("rest", "task"),
        subject_ids=(2,),
        data_reader="eeg",
        subject_identities={"2": {"subject_id": 2}},
        backbone_sha256="backbone-hash",
        preprocessing_hash="preprocess-hash",
        split_identity=features.build_feature_cache_split_identity(
            split_mode="loso",
            train_sessions=("0train",),
            test_session="1test",
            validation_session="1test",
            validation_ratio=None,
            split_seed=None,
        ),
    )
    payload = torch.load(cache_path, weights_only=False)
    assert payload["format_version"] == 1
    assert payload["features"].equal(extracted)
    assert payload["labels"].tolist() == [1, 0, 1]
    assert payload["source_trial_ids"] == [[10], [11], [12]]


def test_feature_helpers_keep_legacy_re_exports() -> None:
    assert runner.extract_frozen_features is features.extract_frozen_features
    assert runner.save_population_feature_cache is features.save_population_feature_cache
    assert linear_head.extract_frozen_features is features.extract_frozen_features
    assert linear_head.feature_cache_dtype_from_name is features.feature_cache_dtype_from_name
