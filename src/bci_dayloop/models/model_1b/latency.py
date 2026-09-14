"""Latency-only orchestration for the 1B encoder; no logits are produced."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from bci_dayloop.benchmarking.core import synchronize_device
from bci_dayloop.preprocessing.canonical import SignalCanonicalizer
from bci_dayloop.runtime.types import PreparedModelInput, RawEEGWindow

from .backbone import Model1BBackbone
from .config import Model1BConfig
from .preprocessing import Model1BInputTransform
from .tokenization import Model1BTokenizer


class Model1BLatencyRuntime:
    """Prepare-only Runtime-shaped object used solely by the realtime bridge."""

    def __init__(self, config: Model1BConfig) -> None:
        self.input_transform = Model1BInputTransform(config)
        self.canonicalizer = SignalCanonicalizer(target_unit="uV")

    @property
    def input_contract(self):
        return self.input_transform.input_contract

    def prepare(self, raw_window: RawEEGWindow) -> PreparedModelInput:
        return self.input_transform.transform(self.canonicalizer.transform(raw_window))


@dataclass(frozen=True, slots=True)
class BackboneLatencyRecord:
    preprocessing_ms: float
    tokenization_ms: float
    encoder_ms: float
    compute_total_ms: float
    embedding_shape: tuple[int, int, int]


class Model1BLatencyRunner:
    """Time raw prepare + tokenization + final encoder embedding exactly once."""

    def __init__(self, config: Model1BConfig, backbone: Model1BBackbone) -> None:
        self.config = config
        self.backbone = backbone
        self.runtime = Model1BLatencyRuntime(config)
        self.tokenizer = Model1BTokenizer(config)
        self.device = backbone.device_object

    def run_one(
        self,
        raw_window: RawEEGWindow,
        *,
        prepare_validator: Callable[[object], None] | None = None,
    ) -> BackboneLatencyRecord:
        synchronize_device(self.device)
        total_started = time.perf_counter()

        started = time.perf_counter()
        prepared = self.runtime.prepare(raw_window)
        synchronize_device(self.device)
        preprocessing_finished = time.perf_counter()
        if prepare_validator is not None:
            prepare_validator(prepared)

        signal = prepared.get_tensor("signal")[0].cpu().numpy()
        channel_mask = prepared.get_tensor("channel_valid_mask")[0].cpu().numpy()
        tokenized = self.tokenizer.tokenize(signal, channel_mask).as_batch(self.device)
        synchronize_device(self.device)
        tokenization_finished = time.perf_counter()

        embeddings = self.backbone(tokenized)
        synchronize_device(self.device)
        encoder_finished = time.perf_counter()
        if tuple(embeddings.shape) != (1, self.config.num_tokens, self.config.d_model):
            raise RuntimeError("1B latency runner received an invalid final encoder embedding")

        return BackboneLatencyRecord(
            preprocessing_ms=(preprocessing_finished - started) * 1000.0,
            tokenization_ms=(tokenization_finished - preprocessing_finished) * 1000.0,
            encoder_ms=(encoder_finished - tokenization_finished) * 1000.0,
            compute_total_ms=(encoder_finished - total_started) * 1000.0,
            embedding_shape=tuple(int(value) for value in embeddings.shape),
        )
