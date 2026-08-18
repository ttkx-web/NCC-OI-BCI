from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Protocol

import numpy as np

from bci_dayloop.packages.loader import LoadedRuntimePackage
from bci_dayloop.runtime.types import RawEEGWindow
from bci_dayloop.serving.profiles import (
    MOTOR_IMAGERY_CLASS_NAMES,
    STEP_SEC,
    DeviceProfile,
    get_device_profile,
)
from bci_dayloop.serving.protocol import (
    ParsedFeedback,
    ParsedWindow,
    PredictionResult,
    ProtocolError,
    hello_message,
    new_observation_id,
)


class ServingBackend(Protocol):
    def hello_payload(self) -> dict[str, Any]: ...

    def predict(self, window: ParsedWindow) -> PredictionResult: ...

    def ack_feedback(self, feedback: ParsedFeedback) -> dict[str, Any]: ...


FIXED_MOCK_PROBABILITIES: tuple[float, ...] = (0.55, 0.20, 0.15, 0.10)


@dataclass(slots=True)
class MockModelBackend:
    profile: DeviceProfile
    class_names: tuple[str, ...] = MOTOR_IMAGERY_CLASS_NAMES
    probabilities: tuple[float, ...] = FIXED_MOCK_PROBABILITIES
    model_revision: str = "mock"
    strategy: str = "none"

    @classmethod
    def from_profile_id(cls, profile_id: str) -> "MockModelBackend":
        return cls(profile=get_device_profile(profile_id))

    def hello_payload(self) -> dict[str, Any]:
        return hello_message(
            service="ncc-dev-mock",
            model_name="dev-mock",
            model_type="mock",
            task="motor_imagery",
            class_names=self.class_names,
            model_revision=self.model_revision,
            strategy=self.strategy,
            window_sec=self.profile.window_sec,
            step_sec=STEP_SEC,
        )

    def predict(self, window: ParsedWindow) -> PredictionResult:
        if window.profile is None:
            raise ProtocolError(
                "unrecognized EEG layout (expected neuracle59 or bcigo32)",
                code="invalid_window",
                request_id=window.request_id,
            )
        if len(self.probabilities) != len(self.class_names):
            raise ProtocolError("mock probability vector is invalid", code="internal_error")
        class_id = int(np.argmax(self.probabilities))
        return PredictionResult(
            request_id=window.request_id,
            observation_id=new_observation_id(),
            window_id=window.window_id,
            segment_id=window.segment_id,
            class_id=class_id,
            class_name=self.class_names[class_id],
            class_names=self.class_names,
            probabilities=self.probabilities,
            confidence=float(self.probabilities[class_id]),
            model_revision=self.model_revision,
            prepare_latency_ms=0.0,
            inference_latency_ms=0.0,
        )

    def ack_feedback(self, feedback: ParsedFeedback) -> dict[str, Any]:
        from bci_dayloop.serving.protocol import feedback_ack_message

        return feedback_ack_message(
            feedback,
            accepted=True,
            model_revision=self.model_revision,
        )


@dataclass(slots=True)
class RuntimePackageBackend:
    package: LoadedRuntimePackage
    strategy: str = "none"
    model_revision: str = "base"

    def hello_payload(self) -> dict[str, Any]:
        return hello_message(
            service="ncc-unified-runtime",
            model_name=self.package.model_name,
            model_type=self.package.model_type,
            task="motor_imagery",
            class_names=self.package.class_names,
            model_revision=self.model_revision,
            strategy=self.strategy,
            window_sec=self.package.window_sec,
            step_sec=self.package.step_sec,
        )

    def predict(self, window: ParsedWindow) -> PredictionResult:
        raw = RawEEGWindow(
            data=window.data,
            channel_names=list(window.channel_names),
            sample_rate=window.sample_rate,
            unit="uV",
            layout="CT",
            start_time_sec=window.start_time_sec,
            window_id=str(window.window_id),
            metadata={
                "source": "passive_bci",
                "segment_id": window.segment_id,
                "profile_id": None if window.profile is None else window.profile.id,
            },
        )
        started = time.perf_counter()
        prepared = self.package.runtime_model.prepare(raw)
        prepare_ms = (time.perf_counter() - started) * 1000.0
        inferred = time.perf_counter()
        output = self.package.runtime_model.predict_prepared(prepared)
        infer_ms = (time.perf_counter() - inferred) * 1000.0
        probabilities = (
            output.probabilities.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
        )
        class_names = self.package.class_names
        if probabilities.size != len(class_names) or not np.isfinite(probabilities).all():
            raise ProtocolError(
                "runtime probability vector is invalid",
                code="internal_error",
                request_id=window.request_id,
            )
        class_id = int(output.predicted_class)
        if not 0 <= class_id < len(class_names):
            raise ProtocolError(
                "runtime predicted_class is out of range",
                code="internal_error",
                request_id=window.request_id,
            )
        return PredictionResult(
            request_id=window.request_id,
            observation_id=new_observation_id(),
            window_id=window.window_id,
            segment_id=window.segment_id,
            class_id=class_id,
            class_name=class_names[class_id],
            class_names=class_names,
            probabilities=tuple(float(value) for value in probabilities),
            confidence=float(output.confidence),
            model_revision=self.model_revision,
            prepare_latency_ms=prepare_ms,
            inference_latency_ms=infer_ms,
        )

    def ack_feedback(self, feedback: ParsedFeedback) -> dict[str, Any]:
        from bci_dayloop.serving.protocol import feedback_ack_message

        return feedback_ack_message(
            feedback,
            accepted=self.strategy == "none",
            reason=None if self.strategy == "none" else "online strategy is not enabled",
            model_revision=self.model_revision,
        )
