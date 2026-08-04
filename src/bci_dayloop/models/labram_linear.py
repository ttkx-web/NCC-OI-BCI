from __future__ import annotations

import hashlib
import json
import logging
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bci_dayloop.models.base import BaseModelAdapter
from bci_dayloop.models.labram_backbone import get_input_chans, labram_base_patch200_200
from bci_dayloop.utils.config import dump_json, dump_yaml, load_yaml, project_root, resolve_path

LOGGER = logging.getLogger(__name__)


class LaBraMLinearAdapter(BaseModelAdapter):
    model_name = "labram-linear"

    def __init__(
        self,
        channel_names: list[str],
        n_classes: int = 4,
        checkpoint: str | Path = "checkpoints/backbones/labram/labram_base.pth",
        device: str = "cuda",
        amp: bool = True,
        freeze_encoder: bool = True,
        embedding_batch_size: int = 4,
        random_init: bool = False,
        n_patches: int = 4,
        encoder: nn.Module | None = None,
    ) -> None:
        self.channel_names = list(channel_names)
        self.n_classes = int(n_classes)
        self.checkpoint = Path(checkpoint)
        self.device = self._resolve_device(device)
        self.amp = bool(amp and self.device.type == "cuda")
        self.freeze_encoder = bool(freeze_encoder)
        self.embedding_batch_size = int(embedding_batch_size)
        self.random_init = bool(random_init)
        self.n_patches = int(n_patches)
        self.input_chans = get_input_chans(self.channel_names).to(self.device)
        self.encoder = encoder or labram_base_patch200_200(
            num_classes=0,
            num_patches_per_channel_input=self.n_patches,
            use_mean_pooling=True,
            use_abs_pos_emb=True,
            init_values=0.1,
        )
        self.embedding_dim = int(getattr(self.encoder, "embed_dim", 200))
        self.encoder.to(self.device)
        self.head = nn.Linear(self.embedding_dim, self.n_classes).to(self.device)
        self._checkpoint_report: dict[str, Any] = {"random_init": self.random_init}
        if not self.random_init and encoder is None:
            self._load_base_checkpoint(self.checkpoint)
        self._freeze_encoder()

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
            return torch.device("cpu")
        return torch.device(requested)

    def _freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not self.freeze_encoder
        self.encoder.eval() if self.freeze_encoder else self.encoder.train()

    def _load_base_checkpoint(self, checkpoint: str | Path) -> None:
        path = resolve_path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(
                "LaBraM Base checkpoint is missing. Expected: "
                f"{path}\nPlace the official labram-base.pth at this path, set model.checkpoint "
                "in the YAML config, or run scripts/smoke_test_labram.py --random-init."
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state: Any = payload
        if isinstance(payload, dict):
            for key in ("model", "module"):
                if key in payload and isinstance(payload[key], dict):
                    state = payload[key]
                    break
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported LaBraM checkpoint structure: {path}")
        if any(str(key).startswith("student.") for key in state):
            state = {str(key)[8:]: value for key, value in state.items() if str(key).startswith("student.")}
        current = self.encoder.state_dict()
        compatible: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        for key, value in state.items():
            if "relative_position_index" in key or key.startswith(("head.", "lm_head.")):
                continue
            if key == "time_embed" and key in current and value.shape != current[key].shape:
                value = value[:, : current[key].shape[1], :]
            if key in current and current[key].shape == value.shape:
                compatible[key] = value
            else:
                skipped.append(key)
        missing, unexpected = self.encoder.load_state_dict(compatible, strict=False)
        if len(compatible) < max(20, int(len(current) * 0.5)):
            raise RuntimeError(
                f"Checkpoint {path} is not compatible with labram_base_patch200_200: "
                f"loaded {len(compatible)}/{len(current)} tensors"
            )
        self.checkpoint = path
        self._checkpoint_report = {
            "random_init": False,
            "loaded_tensors": len(compatible),
            "model_tensors": len(current),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "skipped_keys": skipped,
        }

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16) if self.amp else nullcontext()

    def extract_embeddings(self, X: np.ndarray, cache_path: str | Path | None = None) -> np.ndarray:
        if cache_path is not None and Path(cache_path).exists():
            with np.load(cache_path) as cached:
                return cached["embeddings"].astype(np.float32, copy=False)
        values = np.asarray(X, dtype=np.float32)
        if values.ndim == 3:
            values = values[None, ...]
        if values.ndim != 4 or values.shape[1] != len(self.channel_names) or values.shape[-1] != 200:
            raise ValueError(
                f"Expected [N,{len(self.channel_names)},A,200] in configured channel order, got {values.shape}"
            )
        self.encoder.eval()
        outputs: list[np.ndarray] = []
        loader = DataLoader(torch.from_numpy(values), batch_size=self.embedding_batch_size, shuffle=False)
        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(self.device, non_blocking=self.device.type == "cuda")
                with self._autocast():
                    if hasattr(self.encoder, "forward_features"):
                        features = self.encoder.forward_features(batch, input_chans=self.input_chans)
                    else:
                        features = self.encoder(batch)
                outputs.append(features.float().cpu().numpy())
        embeddings = np.concatenate(outputs).astype(np.float32, copy=False)
        if cache_path is not None:
            target = Path(cache_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, embeddings=embeddings)
        return embeddings

    def _fit_head(
        self,
        train_embeddings: np.ndarray,
        y_train: np.ndarray,
        val_embeddings: np.ndarray,
        y_val: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        patience: int,
    ) -> dict[str, Any]:
        dataset = TensorDataset(torch.from_numpy(train_embeddings), torch.from_numpy(y_train.astype(np.int64)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.head.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        val_x = torch.from_numpy(val_embeddings).to(self.device)
        val_y = torch.from_numpy(y_val.astype(np.int64)).to(self.device)
        best_state = deepcopy(self.head.state_dict())
        best_loss = float("inf")
        best_accuracy = 0.0
        stagnant = 0
        history: list[dict[str, float]] = []
        for epoch in range(int(epochs)):
            self.head.train()
            losses: list[float] = []
            for features, targets in loader:
                features, targets = features.to(self.device), targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.head(features), targets)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            self.head.eval()
            with torch.inference_mode():
                logits = self.head(val_x)
                val_loss = float(criterion(logits, val_y).cpu())
                val_pred = logits.argmax(dim=1)
                val_accuracy = float((val_pred == val_y).float().mean().cpu())
            history.append(
                {"epoch": float(epoch + 1), "train_loss": float(np.mean(losses)), "val_loss": val_loss, "val_accuracy": val_accuracy}
            )
            if val_loss < best_loss - 1e-7:
                best_loss, best_accuracy = val_loss, val_accuracy
                best_state = deepcopy(self.head.state_dict())
                stagnant = 0
            else:
                stagnant += 1
                if stagnant >= patience:
                    break
        self.head.load_state_dict(best_state)
        return {
            "best_val_loss": best_loss,
            "best_val_accuracy": best_accuracy,
            "epochs_ran": len(history),
            "history": history,
        }

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        epochs: int = 80,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 12,
        cache_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        if validation_data is None:
            raise ValueError("validation_data=(X_val, y_val) is required to keep the session split explicit")
        X_val, y_val = validation_data
        cache = Path(cache_dir) if cache_dir is not None else None
        train_embeddings = self.extract_embeddings(X, cache / "train_embeddings.npz" if cache else None)
        val_embeddings = self.extract_embeddings(X_val, cache / "val_embeddings.npz" if cache else None)
        return self._fit_head(
            train_embeddings,
            np.asarray(y),
            val_embeddings,
            np.asarray(y_val),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        embeddings = self.extract_embeddings(X)
        self.head.eval()
        with torch.inference_mode():
            logits = self.head(torch.from_numpy(embeddings).to(self.device))
            return torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)

    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        learning_rate: float = 1e-4,
        epochs: int = 1,
    ) -> dict[str, Any]:
        embeddings = self.extract_embeddings(X)
        targets = np.asarray(y, dtype=np.int64)
        optimizer = torch.optim.AdamW(self.head.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        features_t = torch.from_numpy(embeddings).to(self.device)
        targets_t = torch.from_numpy(targets).to(self.device)
        self.head.train()
        loss_value = 0.0
        for _ in range(int(epochs)):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(self.head(features_t), targets_t)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
        predictions = self.head(features_t).argmax(dim=1).detach().cpu().numpy()
        return {"updated": float(len(targets)), "loss": loss_value, "accuracy": float(accuracy_score(targets, predictions))}

    def save(
        self,
        path: str | Path,
        *,
        preprocessing: dict[str, Any] | None = None,
        label_map: dict[int | str, str] | None = None,
        command_map: dict[str, str] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> Path:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.head.state_dict(),
                "embedding_dim": self.embedding_dim,
                "n_classes": self.n_classes,
            },
            target / "head.pt",
        )
        model_config = {
            "name": self.model_name,
            "architecture": "labram_base_patch200_200",
            "n_classes": self.n_classes,
            "embedding_dim": self.embedding_dim,
            "n_patches": self.n_patches,
            "channel_names": self.channel_names,
            "freeze_encoder": self.freeze_encoder,
            "amp": self.amp,
            "embedding_batch_size": self.embedding_batch_size,
        }
        dump_yaml(model_config, target / "model.yaml")
        dump_yaml(preprocessing or {}, target / "preprocessing.yaml")
        dump_json({str(k): v for k, v in (label_map or {}).items()}, target / "label_map.json")
        dump_json(command_map or {}, target / "command_map.json")
        dump_json(metrics or {}, target / "metrics.json")
        checkpoint_path = resolve_path(self.checkpoint)
        digest = None
        if checkpoint_path.exists():
            sha = hashlib.sha256()
            with checkpoint_path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    sha.update(block)
            digest = sha.hexdigest()
        dump_json(
            {
                "architecture": "labram_base_patch200_200",
                "checkpoint_path": str(self.checkpoint),
                "checkpoint_path_absolute": str(checkpoint_path),
                "checkpoint_sha256": digest,
                "random_init": self.random_init,
                "checkpoint_report": self._checkpoint_report,
            },
            target / "base_model.json",
        )
        return target

    def load(self, path: str | Path) -> "LaBraMLinearAdapter":
        package = Path(path)
        payload = torch.load(package / "head.pt", map_location="cpu", weights_only=True)
        self.head = nn.Linear(int(payload["embedding_dim"]), int(payload["n_classes"])).to(self.device)
        self.head.load_state_dict(payload["state_dict"])
        self.head.eval()
        return self

    @classmethod
    def from_package(cls, path: str | Path, device: str = "cpu") -> "LaBraMLinearAdapter":
        package = Path(path)
        model_config = load_yaml(package / "model.yaml")
        with (package / "base_model.json").open("r", encoding="utf-8") as handle:
            base = json.load(handle)
        checkpoint = Path(base.get("checkpoint_path_absolute", ""))
        if not checkpoint.exists():
            checkpoint = Path(base.get("checkpoint_path", "checkpoints/backbones/labram/labram_base.pth"))
            if not checkpoint.is_absolute():
                candidates = [package / checkpoint, project_root() / checkpoint]
                checkpoint = next((item for item in candidates if item.exists()), candidates[-1])
        adapter = cls(
            channel_names=model_config["channel_names"],
            n_classes=int(model_config["n_classes"]),
            checkpoint=checkpoint,
            device=device,
            amp=bool(model_config.get("amp", True)),
            freeze_encoder=True,
            embedding_batch_size=int(model_config.get("embedding_batch_size", 4)),
            random_init=bool(base.get("random_init", False)),
            n_patches=int(model_config.get("n_patches", 4)),
        )
        return adapter.load(package)
