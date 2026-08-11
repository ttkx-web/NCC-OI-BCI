from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from bci_dayloop.realtime.channel_units import EEG_UNIT
from bci_dayloop.runtime.types import (
    InputContract,
    PreparedModelInput,
)

from .contracts import RealtimeWindow
from .runtime_mapping import (
    APPROVED_NEURACLE_59_TO_STANDARD64,
    ApprovedRealtimeMappingPolicy,
)


class RuntimePrepareOnly(Protocol):
    @property
    def input_contract(self) -> InputContract: ...

    def prepare(self, raw_window: object) -> PreparedModelInput: ...


class RuntimePackageLike(Protocol):
    runtime_model: RuntimePrepareOnly
    model_type: str
    model_name: str
    is_test_head: bool


class RealtimePolicyError(ValueError):
    """A sanitized, pre-device Stage2B compatibility failure."""


@dataclass(frozen=True, slots=True)
class PreparedValidation:
    failure_reason: str | None
    signal_shape: tuple[int, ...] | None
    valid_channel_count: int | None


class RealtimeModelPolicy(ABC):
    model_type: str
    policy_id: str

    @property
    @abstractmethod
    def required_channel_names(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def missing_target_channels(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def ignored_source_channels(self) -> tuple[str, ...]: ...

    @abstractmethod
    def validate_package(self, package: RuntimePackageLike) -> None: ...

    @abstractmethod
    def select_source(
        self,
        window: RealtimeWindow,
    ) -> tuple[np.ndarray, tuple[str, ...]]: ...

    @abstractmethod
    def validate_prepared(
        self,
        prepared: PreparedModelInput,
        runtime_model: RuntimePrepareOnly,
    ) -> PreparedValidation: ...


def _normalized_duplicates(names: Sequence[str]) -> tuple[str, ...]:
    normalized = [str(name).strip().upper() for name in names]
    return tuple(
        sorted({name for name in normalized if normalized.count(name) > 1})
    )


def _validate_exact_unique_channels(
    names: Sequence[str],
    *,
    logical_name: str,
) -> tuple[str, ...]:
    values = tuple(str(name) for name in names)
    if not values:
        raise ValueError(f"{logical_name} cannot be empty")
    if any(not name or name != name.strip() for name in values):
        raise ValueError(
            f"{logical_name} contains an empty or untrimmed channel name"
        )
    duplicates = _normalized_duplicates(values)
    if duplicates:
        raise ValueError(
            f"{logical_name} contains duplicate or alias-ambiguous channels: "
            f"{duplicates}"
        )
    return values


class Model50MRealtimePolicy(RealtimeModelPolicy):
    model_type = "model_50m"
    policy_id = "model_50m_neuracle_59_to_standard64_v1"

    def __init__(
        self,
        *,
        mapping: ApprovedRealtimeMappingPolicy = (
            APPROVED_NEURACLE_59_TO_STANDARD64
        ),
    ) -> None:
        self.mapping = mapping

    @property
    def required_channel_names(self) -> tuple[str, ...]:
        return self.mapping.target_channel_names

    @property
    def missing_target_channels(self) -> tuple[str, ...]:
        return self.mapping.expected_missing_target_channels

    @property
    def ignored_source_channels(self) -> tuple[str, ...]:
        return self.mapping.expected_ignored_source_channels

    def validate_package(self, package: RuntimePackageLike) -> None:
        if package.model_type != self.model_type:
            raise ValueError(
                "Model50MRealtimePolicy requires model_type='model_50m'"
            )
        if package.is_test_head:
            raise ValueError("A test head is not allowed for realtime inference")
        failure = self._runtime_contract_failure(package.runtime_model)
        if failure:
            raise ValueError(failure)

    def select_source(
        self,
        window: RealtimeWindow,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        return window.samples, window.channel_names

    def validate_prepared(
        self,
        prepared: PreparedModelInput,
        runtime_model: RuntimePrepareOnly,
    ) -> PreparedValidation:
        if not isinstance(prepared, PreparedModelInput):
            return PreparedValidation(
                "RuntimeModel.prepare must return PreparedModelInput",
                None,
                None,
            )
        failure = self._runtime_contract_failure(runtime_model)
        if failure:
            return PreparedValidation(failure, None, None)
        contract = runtime_model.input_contract
        if (
            not isinstance(prepared.model_input, dict)
            or not set(contract.model_input_keys).issubset(prepared.model_input)
        ):
            return PreparedValidation(
                "prepared model_input is missing Runtime Package keys",
                None,
                None,
            )
        signal = prepared.model_input["signal"]
        mask = prepared.model_input["channel_valid_mask"]
        if not isinstance(signal, torch.Tensor) or not isinstance(mask, torch.Tensor):
            return PreparedValidation(
                "prepared signal and channel_valid_mask must be tensors",
                None,
                None,
            )
        signal_shape = tuple(int(value) for value in signal.shape)
        if (
            signal.dtype != torch.float32
            or signal_shape != (1, 64, 400)
            or not torch.isfinite(signal).all().item()
        ):
            return PreparedValidation(
                "prepared signal must be finite float32 with shape [1, 64, 400]",
                signal_shape,
                None,
            )
        if tuple(mask.shape) != (1, 64) or not (
            mask.dtype == torch.bool
            or torch.all((mask == 0) | (mask == 1)).item()
        ):
            return PreparedValidation(
                "prepared channel_valid_mask must be [1, 64] containing only 0/1 or bool",
                signal_shape,
                None,
            )
        mask_bool = mask.to(dtype=torch.bool)[0]
        valid_count = int(mask_bool.sum().item())
        if valid_count != 57:
            return PreparedValidation(
                "prepared channel_valid_mask valid count must be 57",
                signal_shape,
                valid_count,
            )
        false_positions = tuple(
            index
            for index, valid in enumerate(mask_bool.tolist())
            if not valid
        )
        if false_positions != self.mapping.target_missing_indices:
            return PreparedValidation(
                "prepared channel_valid_mask false positions do not match the approved policy",
                signal_shape,
                valid_count,
            )
        diagnostics = prepared.diagnostics
        if not isinstance(diagnostics, Mapping):
            return PreparedValidation(
                "prepared diagnostics must be a mapping",
                signal_shape,
                valid_count,
            )
        if int(diagnostics.get("mapped_channel_count", -1)) != 57:
            return PreparedValidation(
                "prepared mapped_channel_count must be 57",
                signal_shape,
                valid_count,
            )
        if int(diagnostics.get("missing_channel_count", -1)) != 7:
            return PreparedValidation(
                "prepared missing_channel_count must be 7",
                signal_shape,
                valid_count,
            )
        unknown = tuple(
            str(value)
            for value in diagnostics.get("unknown_channel_names", ())
        )
        if unknown != self.mapping.expected_ignored_source_channels:
            return PreparedValidation(
                "prepared unknown source channels do not match the approved policy",
                signal_shape,
                valid_count,
            )
        for name in ("duplicate_channel_count", "padded_points", "cropped_points"):
            if int(diagnostics.get(name, -1)) != 0:
                return PreparedValidation(
                    f"prepared {name} must be 0",
                    signal_shape,
                    valid_count,
                )
        return PreparedValidation(None, signal_shape, valid_count)

    def _runtime_contract_failure(
        self,
        runtime_model: RuntimePrepareOnly,
    ) -> str | None:
        contract = runtime_model.input_contract
        if contract.channel_names != self.mapping.target_channel_names:
            return "Runtime Package channel order does not match the approved policy"
        if (
            contract.sample_rate != 100.0
            or contract.window_sec != 4.0
            or contract.num_samples != 400
        ):
            return (
                "Runtime Package must declare 64 channels at 100 Hz "
                "for 4.0 seconds / 400 samples"
            )
        if (
            contract.input_unit != EEG_UNIT
            or contract.model_input_keys
            != ("signal", "channel_valid_mask")
        ):
            return (
                "Runtime Package input unit or model input keys "
                "do not match the approved policy"
            )
        config = getattr(
            getattr(runtime_model, "input_transform", None),
            "config",
            None,
        )
        if (
            config is None
            or getattr(config, "output_layer_idx", None) != 8
            or getattr(config, "aggregation", None) != "flatten"
        ):
            return (
                "Runtime Package must use output_layer_idx=8 "
                "and aggregation=flatten"
            )
        return None


class LaBraMRealtimePolicy(RealtimeModelPolicy):
    model_type = "labram"
    policy_id = "labram_package_required_channels_v1"

    def __init__(self, package: RuntimePackageLike) -> None:
        self._required_channel_names = _validate_exact_unique_channels(
            package.runtime_model.input_contract.channel_names,
            logical_name="LaBraM required channel_names",
        )
        self._source_indices = self._validate_source_mapping()
        self._ignored_source_channels = tuple(
            name
            for name in APPROVED_NEURACLE_59_TO_STANDARD64.source_channel_names
            if name not in self._required_channel_names
        )
        self.validate_package(package)

    @property
    def required_channel_names(self) -> tuple[str, ...]:
        return self._required_channel_names

    @property
    def missing_target_channels(self) -> tuple[str, ...]:
        return ()

    @property
    def ignored_source_channels(self) -> tuple[str, ...]:
        return self._ignored_source_channels

    def _validate_source_mapping(self) -> tuple[int, ...]:
        source = _validate_exact_unique_channels(
            APPROVED_NEURACLE_59_TO_STANDARD64.source_channel_names,
            logical_name="approved Neuracle source channel_names",
        )
        source_index = {name: index for index, name in enumerate(source)}
        source_normalized = {name.upper(): name for name in source}
        missing: list[str] = []
        ambiguous: list[str] = []
        for name in self._required_channel_names:
            if name in source_index:
                continue
            if name.upper() in source_normalized:
                ambiguous.append(name)
            else:
                missing.append(name)
        if ambiguous:
            raise ValueError(
                "LaBraM required channels have alias/case ambiguity: "
                f"{ambiguous}"
            )
        if missing:
            raise ValueError(
                "LaBraM required channels are missing from the approved "
                f"Neuracle source: {missing}"
            )
        return tuple(source_index[name] for name in self._required_channel_names)

    def validate_package(self, package: RuntimePackageLike) -> None:
        if package.model_type != self.model_type:
            raise ValueError(
                "LaBraMRealtimePolicy requires model_type='labram'"
            )
        if package.is_test_head:
            raise ValueError("A test head is not allowed for realtime inference")
        contract = package.runtime_model.input_contract
        if tuple(contract.channel_names) != self._required_channel_names:
            raise ValueError("LaBraM Runtime Package channel contract changed")
        if (
            contract.sample_rate != 200.0
            or contract.window_sec != 4.0
            or contract.num_samples != 800
            or contract.input_unit != EEG_UNIT
            or contract.tensor_layout != "BCTP"
            or contract.model_input_keys != ("signal",)
            or contract.strict_window_duration is not True
        ):
            raise ValueError(
                "LaBraM realtime package must declare 200 Hz, 4.0 seconds, "
                "800 samples, uV, BCTP, and model_input_keys=('signal',)"
            )

    def select_source(
        self,
        window: RealtimeWindow,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        selected = np.asarray(
            window.samples[np.asarray(self._source_indices, dtype=np.int64), :],
            dtype=np.float32,
        )
        expected_shape = (len(self._required_channel_names), 4000)
        if selected.shape != expected_shape:
            raise ValueError(
                "LaBraM explicit source selection produced an unexpected "
                f"shape: {selected.shape} != {expected_shape}"
            )
        return selected, self._required_channel_names

    def validate_prepared(
        self,
        prepared: PreparedModelInput,
        runtime_model: RuntimePrepareOnly,
    ) -> PreparedValidation:
        if not isinstance(prepared, PreparedModelInput):
            return PreparedValidation(
                "RuntimeModel.prepare must return PreparedModelInput",
                None,
                None,
            )
        try:
            self.validate_package(
                _RuntimePackageView(runtime_model=runtime_model)
            )
        except ValueError as exc:
            return PreparedValidation(str(exc), None, None)
        if not isinstance(prepared.model_input, dict):
            return PreparedValidation(
                "LaBraM prepared model_input must be a dict",
                None,
                None,
            )
        keys = tuple(prepared.model_input.keys())
        if set(keys) != {"signal"}:
            return PreparedValidation(
                "LaBraM prepared model_input must contain only the package-approved 'signal' key",
                None,
                None,
            )
        signal = prepared.model_input["signal"]
        if not isinstance(signal, torch.Tensor):
            return PreparedValidation(
                "LaBraM prepared signal must be a tensor",
                None,
                None,
            )
        signal_shape = tuple(int(value) for value in signal.shape)
        expected_shape = (1, len(self._required_channel_names), 4, 200)
        if (
            signal.dtype != torch.float32
            or signal_shape != expected_shape
            or not torch.isfinite(signal).all().item()
        ):
            return PreparedValidation(
                "LaBraM prepared signal must be finite float32 with shape "
                f"{list(expected_shape)}",
                signal_shape,
                None,
            )
        diagnostics = prepared.diagnostics
        if not isinstance(diagnostics, Mapping):
            return PreparedValidation(
                "LaBraM prepared diagnostics must be a mapping",
                signal_shape,
                None,
            )
        if tuple(diagnostics.get("missing_channel_names", ())) != ():
            return PreparedValidation(
                "LaBraM prepared input reports missing channels",
                signal_shape,
                None,
            )
        if int(diagnostics.get("source_channel_count", -1)) != len(
            self._required_channel_names
        ):
            return PreparedValidation(
                "LaBraM prepared source_channel_count must equal required channels",
                signal_shape,
                None,
            )
        if int(diagnostics.get("target_channel_count", -1)) != len(
            self._required_channel_names
        ):
            return PreparedValidation(
                "LaBraM prepared target_channel_count must equal required channels",
                signal_shape,
                None,
            )
        return PreparedValidation(
            None,
            signal_shape,
            len(self._required_channel_names),
        )


@dataclass(frozen=True, slots=True)
class _RuntimePackageView:
    runtime_model: RuntimePrepareOnly
    model_type: str = "labram"
    model_name: str = "labram"
    is_test_head: bool = False


PolicyBuilder = Callable[[RuntimePackageLike], RealtimeModelPolicy]


class RealtimeModelPolicyRegistry:
    _builders: dict[str, PolicyBuilder] = {
        "model_50m": lambda package: _build_model_50m_policy(package),
        "labram": lambda package: LaBraMRealtimePolicy(package),
    }

    @classmethod
    def register(cls, model_type: str, builder: PolicyBuilder) -> None:
        name = str(model_type).strip().lower()
        if not name:
            raise ValueError("realtime policy model_type cannot be empty")
        if name in cls._builders:
            raise ValueError(
                f"Realtime policy for model_type {name!r} is already registered"
            )
        cls._builders[name] = builder

    @classmethod
    def create(cls, package: RuntimePackageLike) -> RealtimeModelPolicy:
        model_type = str(package.model_type).strip().lower()
        builder = cls._builders.get(model_type)
        if builder is None:
            available = ", ".join(sorted(cls._builders))
            raise RealtimePolicyError(
                f"No approved Stage2B realtime policy for model_type "
                f"{model_type!r}. Available: {available}. BLOCKED."
            )
        try:
            return builder(package)
        except RealtimePolicyError:
            raise
        except ValueError as exc:
            raise RealtimePolicyError(
                f"Stage2B realtime compatibility is BLOCKED: {exc}"
            ) from exc

    @classmethod
    def list_model_types(cls) -> list[str]:
        return sorted(cls._builders)


def _build_model_50m_policy(
    package: RuntimePackageLike,
) -> RealtimeModelPolicy:
    policy = Model50MRealtimePolicy()
    policy.validate_package(package)
    return policy
