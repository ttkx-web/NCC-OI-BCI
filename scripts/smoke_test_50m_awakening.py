"""Run one tiny Awakening loader -> preprocessing -> frozen 50M forward smoke test."""

from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from _bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:
    from scripts._bootstrap import ROOT  # noqa: F401

from bci_dayloop.data.dataset_adapter_registry import (
    DEFAULT_DATASET_ADAPTER_REGISTRY,
    LegacyHDF5Adapter,
    inspect_hdf5_dataset,
)
from bci_dayloop.data.dataset_registry import AWAKENING_2S
from bci_dayloop.models.model_50m.backbone import Model50MBackbone
from bci_dayloop.models.model_50m.classifier import Model50MClassifier
from bci_dayloop.models.model_50m.preprocessing import Model50MPreprocessor
from bci_dayloop.models.model_50m.tokenization import (
    Model50MTokenizer,
    stack_model50m_tokens,
)
from bci_dayloop.training.model_50m.awakening import (
    awakening_model_config,
    validate_awakening_dataset,
)
from bci_dayloop.training.model_50m.linear_head import resolve_repo_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=AWAKENING_2S.default_data_root)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--session", choices=("S1", "S2"), default="S1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--backbone-checkpoint",
        default="checkpoints/backbones/50m/model_deploy.pt",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    return parser


def _small_hdf5_batch(path, *, session: str, batch_size: int):
    with h5py.File(path, "r") as handle:
        sessions = handle["session_ids"].asstr()[:]
        indices = np.flatnonzero(sessions == session)[:batch_size]
        if len(indices) != batch_size:
            raise ValueError(f"Not enough {session} rows for smoke batch {batch_size}.")
        data = handle["data"][indices].astype(np.float32, copy=False)
        labels = handle["labels"][indices].astype(np.int64, copy=False)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data), torch.from_numpy(labels)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return next(iter(loader))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.batch_size > 4:
        raise ValueError("Smoke --batch-size must be in [1,4].")
    audit = validate_awakening_dataset(
        args.data_root, subjects=[args.subject], require_manifests=True
    )
    path = audit.subject_paths[args.subject]
    adapter = DEFAULT_DATASET_ADAPTER_REGISTRY.resolve(inspect_hdf5_dataset(path))
    if not isinstance(adapter, LegacyHDF5Adapter):
        raise RuntimeError("Awakening did not select LegacyHDF5Adapter.")
    raw, labels = _small_hdf5_batch(
        path, session=args.session, batch_size=args.batch_size
    )
    if tuple(raw.shape) != (args.batch_size, 62, 400):
        raise RuntimeError(f"Unexpected raw batch shape: {tuple(raw.shape)}")

    config = awakening_model_config(
        checkpoint=args.backbone_checkpoint, device=args.device
    )
    preprocessor = Model50MPreprocessor(config)
    tokenizer = Model50MTokenizer(config)
    processed = [
        preprocessor(
            signal=sample.numpy(),
            channel_names=AWAKENING_2S.source_channels,
            original_sample_rate=200.0,
            input_unit="uV",
        )
        for sample in raw
    ]
    if {item.shape for item in processed} != {(64, 200)}:
        raise RuntimeError("Awakening preprocessing did not produce [64,200].")
    if {item.mapped_channel_count for item in processed} != {60}:
        raise RuntimeError("Expected exactly 60 mapped channels.")
    if {item.missing_channel_count for item in processed} != {4}:
        raise RuntimeError("Expected exactly four missing model channels.")
    if any(item.padded_points or item.cropped_points for item in processed):
        raise RuntimeError("Temporal padding/cropping is forbidden for Awakening 2s.")
    unknown = set().union(*(set(item.unknown_channel_names) for item in processed))
    if unknown != {"TP9", "TP10"}:
        raise RuntimeError(f"Unexpected ignored source channels: {sorted(unknown)}")

    tokens = [tokenizer(item) for item in processed]
    model_batch = stack_model50m_tokens(tokens, device=config.device)
    if model_batch.num_tokens != 128:
        raise RuntimeError(f"Expected 128 tokens, got {model_batch.num_tokens}.")
    backbone = Model50MBackbone(config=config, load_checkpoint=True, freeze=True)
    classifier = Model50MClassifier(config=config, backbone=backbone)
    classifier.eval()
    with torch.no_grad():
        features = classifier.extract_features(model_batch)
        logits = classifier.head(features)
    if tuple(features.shape) != (args.batch_size, 65536):
        raise RuntimeError(f"Unexpected feature shape: {tuple(features.shape)}")
    if tuple(logits.shape) != (args.batch_size, 2):
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

    result = {
        "adapter": adapter.name,
        "raw_batch_shape": list(raw.shape),
        "labels_shape": list(labels.shape),
        "model_input_shape": [args.batch_size, 64, 200],
        "resample_count_per_sample": 1,
        "temporal_padded_points": 0,
        "temporal_cropped_points": 0,
        "mapped_channels": 60,
        "ignored_source_channels": ["TP9", "TP10"],
        "missing_model_channels": ["AF7", "AF8", "F9", "F10"],
        "token_count": model_batch.num_tokens,
        "flatten_feature_shape": list(features.shape),
        "logits_shape": list(logits.shape),
        "class_order": list(AWAKENING_2S.class_names),
        "backbone_checkpoint": str(resolve_repo_path(args.backbone_checkpoint)),
        "optimizer_steps": 0,
        "writes": 0,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
