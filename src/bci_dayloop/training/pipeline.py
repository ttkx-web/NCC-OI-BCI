from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.data.preprocessing import EEGPreprocessor
from bci_dayloop.models.factory import ModelFactory
from bci_dayloop.utils.config import resolve_path, seed_everything
from bci_dayloop.utils.metrics import classification_metrics


def train_linear_probe(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    root = resolve_path(".")
    seed = int(config["project"].get("seed", 42))
    seed_everything(seed)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    run_dir = resolve_path(config["project"]["run_dir"], root)
    package_dir = run_dir / "model_package"
    cache_dir = run_dir / "embedding_cache"
    dataset = EEGHDF5(resolve_path(data_config["output_hdf5"], root))
    metadata = dataset.metadata
    train_session = dataset.load(str(data_config["train_session"]))
    test_session = dataset.load(str(data_config["test_session"]))
    preprocessor = EEGPreprocessor(config["preprocessing"])
    X_all = preprocessor.transform(train_session["data"], metadata.sample_rate, metadata.unit)
    X_test = preprocessor.transform(test_session["data"], metadata.sample_rate, metadata.unit)
    indices = np.arange(len(X_all))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=float(training_config.get("validation_fraction", 0.2)),
        stratify=train_session["labels"],
        random_state=seed,
    )
    adapter = ModelFactory.create(
        str(model_config["name"]),
        channel_names=metadata.channel_names,
        n_classes=len(metadata.class_names),
        checkpoint=resolve_path(model_config["checkpoint"], root),
        device=str(model_config.get("device", "cuda")),
        amp=bool(model_config.get("amp", True)),
        freeze_encoder=bool(model_config.get("freeze_encoder", True)),
        embedding_batch_size=int(model_config.get("embedding_batch_size", 4)),
        random_init=bool(model_config.get("random_init", False)),
        n_patches=X_all.shape[2],
    )
    fit_metrics = adapter.fit(
        X_all[train_idx],
        train_session["labels"][train_idx],
        validation_data=(X_all[val_idx], train_session["labels"][val_idx]),
        epochs=int(training_config.get("epochs", 80)),
        batch_size=int(training_config.get("batch_size", 64)),
        learning_rate=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
        patience=int(training_config.get("patience", 12)),
        cache_dir=cache_dir,
    )
    test_cache = cache_dir / "test_embeddings.npz"
    test_embeddings = adapter.extract_embeddings(X_test, test_cache)
    adapter.head.eval()
    import torch

    with torch.inference_mode():
        logits = adapter.head(torch.from_numpy(test_embeddings).to(adapter.device))
        test_predictions = logits.argmax(dim=1).cpu().numpy()
    metrics = {
        "train_session": str(data_config["train_session"]),
        "test_session": str(data_config["test_session"]),
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_test": int(len(X_test)),
        "fit": fit_metrics,
        "test": classification_metrics(test_session["labels"], test_predictions),
    }
    label_map = {index: name for index, name in enumerate(metadata.class_names)}
    command_map = dict(config["labels"]["command_map"])
    adapter.save(
        package_dir,
        preprocessing=preprocessor.config.to_dict(),
        label_map=label_map,
        command_map=command_map,
        metrics=metrics,
    )
    return package_dir, metrics

