from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.data.preprocessing import EEGPreprocessor
from bci_dayloop.models.factory import ModelFactory
from bci_dayloop.utils.config import (
    dump_json,
    resolve_path,
    seed_everything,
)
from bci_dayloop.utils.metrics import classification_metrics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def atomic_torch_save(
    payload: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(payload, temporary)
    temporary.replace(path)


def train_labram_linear_head(
    config: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """
    冻结 LaBraM Encoder，只训练线性分类头。

    本函数只保存 head.pt 和训练指标，不导出 Runtime Package。
    """

    root = resolve_path(".")

    project_config = config["project"]
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    artifact_config = config["artifacts"]

    seed = int(
        project_config.get("seed", 42)
    )
    seed_everything(seed)

    data_path = resolve_path(
        data_config["output_hdf5"],
        root,
    )

    backbone_path = resolve_path(
        model_config["checkpoint"],
        root,
    )

    head_path = resolve_path(
        artifact_config["head_path"],
        root,
    )

    metrics_path = resolve_path(
        artifact_config[
            "training_metrics_path"
        ],
        root,
    )

    cache_dir = resolve_path(
        artifact_config[
            "embedding_cache_dir"
        ],
        root,
    )

    if not data_path.is_file():
        raise FileNotFoundError(
            f"HDF5 data was not found: {data_path}"
        )

    if not backbone_path.is_file():
        raise FileNotFoundError(
            "LaBraM backbone checkpoint was not found: "
            f"{backbone_path}"
        )

    dataset = EEGHDF5(data_path)
    metadata = dataset.metadata

    train_session_name = str(
        data_config["train_session"]
    )
    test_session_name = str(
        data_config["test_session"]
    )

    train_session = dataset.load(
        train_session_name
    )
    test_session = dataset.load(
        test_session_name
    )

    preprocessor = EEGPreprocessor(
        config["preprocessing"]
    )

    X_all = preprocessor.transform(
        train_session["data"],
        metadata.sample_rate,
        metadata.unit,
    )

    X_test = preprocessor.transform(
        test_session["data"],
        metadata.sample_rate,
        metadata.unit,
    )

    if X_all.ndim != 4:
        raise ValueError(
            "Expected preprocessed LaBraM train data "
            f"[N,C,A,200], got {X_all.shape}."
        )

    if X_test.ndim != 4:
        raise ValueError(
            "Expected preprocessed LaBraM test data "
            f"[N,C,A,200], got {X_test.shape}."
        )

    n_patches = int(X_all.shape[2])

    if X_all.shape[-1] != 200:
        raise ValueError(
            "LaBraM patch length must be 200, "
            f"got {X_all.shape[-1]}."
        )

    indices = np.arange(len(X_all))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=float(
            training_config.get(
                "validation_fraction",
                0.2,
            )
        ),
        stratify=train_session["labels"],
        random_state=seed,
    )

    class_names = tuple(
        str(name)
        for name in metadata.class_names
    )

    adapter = ModelFactory.create(
        str(model_config["name"]),
        channel_names=list(
            metadata.channel_names
        ),
        n_classes=len(class_names),
        checkpoint=backbone_path,
        device=str(
            model_config.get(
                "device",
                "cuda",
            )
        ),
        amp=bool(
            model_config.get(
                "amp",
                True,
            )
        ),
        freeze_encoder=True,
        embedding_batch_size=int(
            model_config.get(
                "embedding_batch_size",
                4,
            )
        ),
        random_init=bool(
            model_config.get(
                "random_init",
                False,
            )
        ),
        n_patches=n_patches,
    )

    fit_metrics = adapter.fit(
        X_all[train_idx],
        train_session["labels"][train_idx],
        validation_data=(
            X_all[val_idx],
            train_session["labels"][val_idx],
        ),
        epochs=int(
            training_config.get(
                "epochs",
                80,
            )
        ),
        batch_size=int(
            training_config.get(
                "batch_size",
                64,
            )
        ),
        learning_rate=float(
            training_config.get(
                "learning_rate",
                1e-3,
            )
        ),
        weight_decay=float(
            training_config.get(
                "weight_decay",
                1e-4,
            )
        ),
        patience=int(
            training_config.get(
                "patience",
                12,
            )
        ),
        cache_dir=cache_dir,
    )

    test_cache = (
        cache_dir / "test_embeddings.npz"
    )

    test_embeddings = (
        adapter.extract_embeddings(
            X_test,
            test_cache,
        )
    )

    adapter.head.eval()

    with torch.inference_mode():
        logits = adapter.head(
            torch.from_numpy(
                test_embeddings
            ).to(adapter.device)
        )

        test_predictions = (
            logits.argmax(dim=1)
            .cpu()
            .numpy()
        )

    test_metrics = classification_metrics(
        test_session["labels"],
        test_predictions,
    )

    window_seconds = (
        n_patches
        * preprocessor.config.patch_samples
        / preprocessor.config.target_sample_rate
    )

    metrics = {
        "train_session": train_session_name,
        "test_session": test_session_name,
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_test": int(len(X_test)),
        "model_selection": fit_metrics,
        "final_test": test_metrics,
    }

    head_metadata = {
        "model_type": "labram",
        "model_name": "labram-linear",
        "architecture": str(
            model_config.get(
                "architecture",
                "labram_base_patch200_200",
            )
        ),

        "embedding_dim": int(
            adapter.embedding_dim
        ),
        "num_classes": int(
            adapter.n_classes
        ),
        "class_names": list(class_names),

        "channel_names": [
            str(name)
            for name in metadata.channel_names
        ],

        "n_patches": n_patches,
        "patch_samples": int(
            preprocessor.config.patch_samples
        ),
        "target_sample_rate": float(
            preprocessor.config
            .target_sample_rate
        ),
        "window_seconds": float(
            window_seconds
        ),

        "preprocessing": (
            preprocessor.config.to_dict()
        ),

        "dataset": str(
            metadata.dataset_name
        ),
        "subject": int(
            data_config["subject"]
        ),
        "train_session": train_session_name,
        "test_session": test_session_name,

        "seed": seed,
        "backbone_checkpoint": str(
            backbone_path
        ),
        "backbone_sha256": sha256_file(
            backbone_path
        ),

        "freeze_encoder": True,
        "trained_head": True,
        "is_test_head": False,

        "model_selection": fit_metrics,
        "final_test": test_metrics,
    }

    checkpoint_payload = {
        "format_version": 1,
        "state_dict": (
            adapter.head.state_dict()
        ),
        "metadata": head_metadata,
    }

    atomic_torch_save(
        checkpoint_payload,
        head_path,
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dump_json(
        metrics,
        metrics_path,
    )

    return head_path, metrics