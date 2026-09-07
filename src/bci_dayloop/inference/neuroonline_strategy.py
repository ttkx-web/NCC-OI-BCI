from __future__ import annotations

import copy
import time

from collections import (
    Counter,
    OrderedDict,
    deque,
)
from dataclasses import (
    asdict,
    dataclass,
)
from numbers import Integral

import torch
import torch.nn.functional as F

from bci_dayloop.inference.neuroonline_forward import (
    NeuroOnlineForward,
    build_neuroonline_forward,
)
from bci_dayloop.inference.online_base import (
    OnlineAdaptationStrategy,
)
from bci_dayloop.models.online_features import (
    OnlineTrainableFeatureBackend,
)
from bci_dayloop.runtime.adaptation_types import (
    AdaptationContext,
    FeedbackEvent,
    OnlineObservation,
    OnlineUpdateResult,
)
from bci_dayloop.runtime.model import (
    RuntimeModel,
)
from bci_dayloop.runtime.types import (
    ModelOutput,
    ModelTensor,
    PreparedModelInput,
)


VALID_UPDATE_SCOPES = (
    "generator_and_head",
    "generator_only",
    "head_only",
)

VALID_UPDATE_TRIGGERS = (
    "feedback_count",
    "class_coverage",
)

VALID_MEMORY_STRATEGIES = (
    "recent_fifo",
    "class_balanced_history",
)


@dataclass(frozen=True, slots=True)
class NeuroOnlineConfig:
    """
    NeuroOnline V1 配置。

    V1 默认：
        - 冻结 backbone；
        - 更新 Generator；
        - 更新 classification head；
        - 只接受真实类别标签；
        - 不支持 reward-only；
        - 不使用伪标签。
    """

    # Generator 结构
    num_subject_codes: int = 32
    num_attention_heads: int = 4
    dropout: float = 0.1

    # 在线更新参数
    update_scope: str = "generator_and_head"
    update_trigger: str = "feedback_count"
    min_feedback_per_class: int = 1
    memory_strategy: str = "recent_fifo"
    balanced_memory_per_class: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # 收到至少多少个真实标签后，
    # 才允许进行第一次更新。
    warmup_feedback: int = 32

    # 第一次更新后，每收到多少个新标签，
    # 触发一次更新。
    update_interval: int = 16

    # 只保留最近多少个有标签样本。
    recent_buffer_size: int = 64

    # 一次更新时的小批量大小。
    batch_size: int = 16

    # 每次触发更新时，在 recent buffer
    # 上训练多少轮。
    epochs_per_update: int = 1

    # 尚未收到反馈的 observation 最多保存多少个。
    max_pending_observations: int = 256

    # Generator 初始化和 batch shuffle 随机种子。
    seed: int = 42

    def __post_init__(self) -> None:
        if self.update_scope not in VALID_UPDATE_SCOPES:
            raise ValueError(
                "update_scope must be one of "
                f"{VALID_UPDATE_SCOPES}, got {self.update_scope!r}."
            )

        if self.update_trigger not in VALID_UPDATE_TRIGGERS:
            raise ValueError(
                "update_trigger must be one of "
                f"{VALID_UPDATE_TRIGGERS}, got {self.update_trigger!r}."
            )

        if self.memory_strategy not in VALID_MEMORY_STRATEGIES:
            raise ValueError(
                "memory_strategy must be one of "
                f"{VALID_MEMORY_STRATEGIES}, got {self.memory_strategy!r}."
            )

        if self.min_feedback_per_class <= 0:
            raise ValueError("min_feedback_per_class must be positive.")

        if self.balanced_memory_per_class <= 0:
            raise ValueError("balanced_memory_per_class must be positive.")

        if self.num_subject_codes <= 0:
            raise ValueError(
                "num_subject_codes must be positive."
            )

        if self.num_attention_heads <= 0:
            raise ValueError(
                "num_attention_heads must be positive."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout must be in [0,1)."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay must be non-negative."
            )

        if self.max_grad_norm <= 0:
            raise ValueError(
                "max_grad_norm must be positive."
            )

        if self.warmup_feedback <= 0:
            raise ValueError(
                "warmup_feedback must be positive."
            )

        if self.update_interval <= 0:
            raise ValueError(
                "update_interval must be positive."
            )

        if self.recent_buffer_size <= 0:
            raise ValueError(
                "recent_buffer_size must be positive."
            )

        if (
            self.recent_buffer_size
            < self.warmup_feedback
        ):
            raise ValueError(
                "recent_buffer_size must be greater "
                "than or equal to warmup_feedback."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if self.epochs_per_update <= 0:
            raise ValueError(
                "epochs_per_update must be positive."
            )

        if self.max_pending_observations <= 0:
            raise ValueError(
                "max_pending_observations must be "
                "positive."
            )


@dataclass(slots=True)
class _BufferedSample:
    """
    已经获得真实标签、可以参与在线训练的样本。
    """

    observation_id: str
    model_input: ModelTensor
    label: int


def _clone_model_input_to_cpu(
    model_input: ModelTensor,
) -> ModelTensor:
    """
    将 PreparedModelInput 中的 Tensor 分离并复制到 CPU。

    不能直接长期保存原 Tensor，否则可能：
        - 持有旧计算图；
        - 长期占用 GPU 显存；
        - 被外部代码原地修改。
    """

    if isinstance(
        model_input,
        torch.Tensor,
    ):
        return (
            model_input
            .detach()
            .to(device="cpu")
            .clone()
        )

    if isinstance(
        model_input,
        dict,
    ):
        if not model_input:
            raise ValueError(
                "Cannot buffer an empty model input "
                "dictionary."
            )

        copied: dict[str, torch.Tensor] = {}

        for key, value in model_input.items():
            if not isinstance(key, str):
                raise TypeError(
                    "Model input keys must be strings."
                )

            if not isinstance(
                value,
                torch.Tensor,
            ):
                raise TypeError(
                    f"Model input {key!r} must be "
                    "torch.Tensor."
                )

            copied[key] = (
                value
                .detach()
                .to(device="cpu")
                .clone()
            )

        return copied

    raise TypeError(
        "Unsupported model input type "
        f"{type(model_input).__name__}."
    )


def _validate_single_sample_input(
    model_input: ModelTensor,
) -> None:
    """
    observation 对应一次单窗口预测，
    所以 model_input 的 batch size 必须为 1。
    """

    if isinstance(
        model_input,
        torch.Tensor,
    ):
        if model_input.ndim <= 0:
            raise ValueError(
                "Buffered Tensor must contain a "
                "batch dimension."
            )

        if model_input.shape[0] != 1:
            raise ValueError(
                "One observation must contain exactly "
                "one model input, got batch_size="
                f"{model_input.shape[0]}."
            )

        return

    if isinstance(
        model_input,
        dict,
    ):
        if not model_input:
            raise ValueError(
                "Buffered model input dictionary "
                "cannot be empty."
            )

        for key, value in model_input.items():
            if not isinstance(
                value,
                torch.Tensor,
            ):
                raise TypeError(
                    f"Model input {key!r} must be "
                    "torch.Tensor."
                )

            if value.ndim <= 0:
                raise ValueError(
                    f"Model input {key!r} must contain "
                    "a batch dimension."
                )

            if value.shape[0] != 1:
                raise ValueError(
                    f"Model input {key!r} must have "
                    "batch_size=1, got "
                    f"{value.shape[0]}."
                )

        return

    raise TypeError(
        "Unsupported model input type "
        f"{type(model_input).__name__}."
    )


def _stack_model_inputs(
    model_inputs: list[ModelTensor],
) -> ModelTensor:
    """
    将多个 batch_size=1 的输入拼成训练 batch。

    Tensor：
        [1,...] × B -> [B,...]

    dictionary：
        每个 key 分别在 dim=0 拼接。
    """

    if not model_inputs:
        raise ValueError(
            "Cannot stack an empty model input list."
        )

    first = model_inputs[0]

    if isinstance(
        first,
        torch.Tensor,
    ):
        tensors: list[torch.Tensor] = []

        for value in model_inputs:
            if not isinstance(
                value,
                torch.Tensor,
            ):
                raise TypeError(
                    "Cannot mix Tensor and dictionary "
                    "model inputs in one batch."
                )

            _validate_single_sample_input(value)
            tensors.append(value)

        return torch.cat(
            tensors,
            dim=0,
        )

    if isinstance(
        first,
        dict,
    ):
        expected_keys = set(
            first.keys()
        )

        batched: dict[str, torch.Tensor] = {}

        for value in model_inputs:
            if not isinstance(value, dict):
                raise TypeError(
                    "Cannot mix Tensor and dictionary "
                    "model inputs in one batch."
                )

            if set(value.keys()) != expected_keys:
                raise ValueError(
                    "Buffered model input dictionaries "
                    "do not have identical keys."
                )

            _validate_single_sample_input(value)

        for key in sorted(expected_keys):
            batched[key] = torch.cat(
                [
                    value[key]
                    for value in model_inputs
                    if isinstance(value, dict)
                ],
                dim=0,
            )

        return batched

    raise TypeError(
        "Unsupported model input type "
        f"{type(first).__name__}."
    )


class NeuroOnlineStrategy(
    OnlineAdaptationStrategy
):
    """
    NeuroOnline V1 在线监督适配策略。

    每个在线 session 创建一个实例。

    V1 默认更新范围：
        backbone: frozen
        Generator: trainable
        classifier head: trainable

    诊断消融可通过 update_scope 仅更新 Generator 或 head；
    backbone 始终冻结。
    """

    def __init__(
        self,
        config: NeuroOnlineConfig
        | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else NeuroOnlineConfig()
        )

        self._runtime_model: (
            RuntimeModel | None
        ) = None

        self._context: (
            AdaptationContext | None
        ) = None

        self._forward_model: (
            NeuroOnlineForward | None
        ) = None

        self._optimizer: (
            torch.optim.Optimizer | None
        ) = None

        self._head_parameters: list[
            torch.nn.Parameter
        ] = []

        self._trainable_parameters: list[
            torch.nn.Parameter
        ] = []

        self._generator_parameters: list[
            torch.nn.Parameter
        ] = []

        self._backbone_parameters: list[
            torch.nn.Parameter
        ] = []

        self._parameter_audit: dict[str, int | str] = {}

        # 已预测但尚未收到真实标签的样本。
        self._pending_observations: (
            OrderedDict[str, ModelTensor]
        ) = OrderedDict()

        # 已经收到真实标签的最近样本。
        self._training_buffer: deque[
            _BufferedSample
        ] = deque(
            maxlen=(
                self.config
                .recent_buffer_size
            )
        )

        self._balanced_training_buffers: dict[
            int,
            deque[_BufferedSample],
        ] = {}
        self._label_counts_seen: Counter[int] = Counter()
        self._feedback_count_total = 0
        self._num_classes = 0
        self._latest_feedback_evaluation_ordinal: int | None = None
        self._first_update_evaluation_ordinal: int | None = None
        self._first_update_label_histogram: dict[str, int] | None = None

        self._feedback_since_update = 0
        self._update_step = 0
        self._model_revision = (
            "neuroonline-0"
        )

    @property
    def name(self) -> str:
        return "neuroonline"

    @property
    def initialized(self) -> bool:
        return (
            self._runtime_model is not None
            and self._forward_model is not None
            and self._optimizer is not None
        )

    @property
    def forward_model(
        self,
    ) -> NeuroOnlineForward:
        if self._forward_model is None:
            raise RuntimeError(
                "NeuroOnlineStrategy has not been "
                "initialized."
            )

        return self._forward_model

    @property
    def generator(
        self,
    ) -> torch.nn.Module:
        return self.forward_model.generator

    @property
    def update_step(self) -> int:
        return self._update_step

    @property
    def model_revision(self) -> str:
        return self._model_revision

    @property
    def buffered_sample_count(self) -> int:
        if self.config.memory_strategy == "recent_fifo":
            return len(self._training_buffer)
        return sum(len(items) for items in self._balanced_training_buffers.values())

    @property
    def pending_observation_count(
        self,
    ) -> int:
        return len(
            self._pending_observations
        )

    def _label_histogram(
        self,
        samples: list[_BufferedSample] | None = None,
    ) -> dict[str, int]:
        counts = Counter(
            sample.label
            for sample in (samples if samples is not None else [])
        )
        return {
            str(label): int(counts[label])
            for label in range(self._num_classes)
        }

    def _seen_label_histogram(self) -> dict[str, int]:
        return {
            str(label): int(self._label_counts_seen[label])
            for label in range(self._num_classes)
        }

    def _memory_label_histogram(self) -> dict[str, int]:
        if self.config.memory_strategy == "recent_fifo":
            return self._label_histogram(list(self._training_buffer))
        return {
            str(label): len(self._balanced_training_buffers[label])
            for label in range(self._num_classes)
        }

    def _class_coverage_satisfied(self) -> bool:
        return self._num_classes > 0 and all(
            self._label_counts_seen[label] >= self.config.min_feedback_per_class
            for label in range(self._num_classes)
        )

    def _select_update_samples(self) -> list[_BufferedSample]:
        if self.config.memory_strategy == "recent_fifo":
            return list(self._training_buffer)

        available_labels = [
            label
            for label in range(self._num_classes)
            if self._balanced_training_buffers[label]
        ]
        if not available_labels:
            return []
        n_per_class = min(
            self.config.balanced_memory_per_class,
            min(
                len(self._balanced_training_buffers[label])
                for label in available_labels
            ),
        )
        selected: list[_BufferedSample] = []
        for label in available_labels:
            selected.extend(list(self._balanced_training_buffers[label])[-n_per_class:])
        return selected

    def _telemetry(
        self,
        *,
        trigger_reason: str,
        trigger_satisfied: bool,
        update_applied: bool,
        selected_samples: list[_BufferedSample] | None = None,
    ) -> dict[str, object]:
        selected = selected_samples or []
        memory_histogram = self._memory_label_histogram()
        available_memory_classes = [
            label
            for label in range(self._num_classes)
            if memory_histogram[str(label)] > 0
        ]
        missing_memory_classes = [
            label
            for label in range(self._num_classes)
            if memory_histogram[str(label)] == 0
        ]
        return {
            "update_trigger": self.config.update_trigger,
            "memory_strategy": self.config.memory_strategy,
            "trigger_satisfied": trigger_satisfied,
            "feedback_count_total": self._feedback_count_total,
            "model_class_count": self._num_classes,
            "label_counts_seen": self._seen_label_histogram(),
            "class_coverage_satisfied": self._class_coverage_satisfied(),
            "min_feedback_per_class": self.config.min_feedback_per_class,
            "available_memory_classes": available_memory_classes,
            "missing_memory_classes": missing_memory_classes,
            "memory_label_histogram": memory_histogram,
            "selected_update_label_histogram": self._label_histogram(selected),
            "selected_update_sample_count": len(selected),
            "trigger_reason": trigger_reason,
            "update_applied": update_applied,
            "update_step": self._update_step,
            "first_update_evaluation_ordinal": self._first_update_evaluation_ordinal,
            "first_update_label_histogram": self._first_update_label_histogram,
        }

    @property
    def parameter_audit(self) -> dict[str, int | str]:
        """Return the audited optimizer/freeze boundary for this run."""

        self._require_initialized()
        return dict(self._parameter_audit)

    def _apply_parameter_scope(self) -> None:
        train_generator = self.config.update_scope in (
            "generator_and_head",
            "generator_only",
        )
        train_head = self.config.update_scope in (
            "generator_and_head",
            "head_only",
        )
        for parameter in self._generator_parameters:
            parameter.requires_grad_(train_generator)
        for parameter in self._head_parameters:
            parameter.requires_grad_(train_head)

    @staticmethod
    def _parameter_count(parameters: list[torch.nn.Parameter]) -> int:
        return sum(int(parameter.numel()) for parameter in parameters)

    @staticmethod
    def _discover_backend_parameters(backend: object) -> list[torch.nn.Parameter]:
        """Discover backend-owned module parameters without changing modes."""

        discovered: dict[int, torch.nn.Parameter] = {}
        visited: set[int] = set()

        def visit(value: object, depth: int) -> None:
            if depth > 4 or id(value) in visited:
                return
            visited.add(id(value))
            if isinstance(value, torch.nn.Module):
                for parameter in value.parameters():
                    discovered[id(parameter)] = parameter
                return
            if isinstance(value, dict):
                for item in value.values():
                    visit(item, depth + 1)
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item, depth + 1)
                return
            attributes = getattr(value, "__dict__", None)
            if isinstance(attributes, dict):
                for item in attributes.values():
                    visit(item, depth + 1)

        visit(backend, 0)
        return list(discovered.values())

    def _validate_parameter_scope(self) -> None:
        generator_count = self._parameter_count(
            [item for item in self._generator_parameters if item.requires_grad]
        )
        head_count = self._parameter_count(
            [item for item in self._head_parameters if item.requires_grad]
        )
        optimizer_count = self._parameter_count(self._trainable_parameters)
        backbone_count = self._parameter_count(
            [item for item in self._backbone_parameters if item.requires_grad]
        )
        expected_generator = self.config.update_scope != "head_only"
        expected_head = self.config.update_scope != "generator_only"
        if (generator_count > 0) != expected_generator:
            raise RuntimeError("NeuroOnline Generator freeze audit failed.")
        if (head_count > 0) != expected_head:
            raise RuntimeError("NeuroOnline classification-head freeze audit failed.")
        if optimizer_count != generator_count + head_count:
            raise RuntimeError("NeuroOnline optimizer parameter audit failed.")
        if backbone_count != 0:
            raise RuntimeError("NeuroOnline backbone freeze audit failed.")

        # All supported backends guarantee this boundary through
        # set_online_mode(..., train_backbone=False). The online optimizer is
        # assembled exclusively from the audited Generator/head lists.
        self._parameter_audit = {
            "update_scope": self.config.update_scope,
            "backbone_trainable_param_count": backbone_count,
            "generator_trainable_param_count": generator_count,
            "head_trainable_param_count": head_count,
            "optimizer_param_count": optimizer_count,
        }

    def _require_initialized(
        self,
    ) -> None:
        if not self.initialized:
            raise RuntimeError(
                "NeuroOnlineStrategy must be "
                "initialized before use."
            )

    def initialize(
        self,
        *,
        runtime_model: RuntimeModel,
        context: AdaptationContext,
    ) -> None:
        """
        每个在线 session 调用一次。

        在这里：
            1. 检查 backend；
            2. 创建 Generator；
            3. 创建 NeuroOnlineForward；
            4. 获取 head 参数；
            5. 创建 optimizer。
        """

        if self.initialized:
            raise RuntimeError(
                "NeuroOnlineStrategy has already "
                "been initialized."
            )

        backend = runtime_model.backend

        if not isinstance(
            backend,
            OnlineTrainableFeatureBackend,
        ):
            raise TypeError(
                "Runtime backend does not support "
                "NeuroOnline adaptation: "
                f"{type(backend).__name__}."
            )

        self._num_classes = int(backend.num_classes)
        if self._num_classes <= 0:
            raise ValueError("NeuroOnline backend num_classes must be positive.")
        self._balanced_training_buffers = {
            label: deque(maxlen=self.config.balanced_memory_per_class)
            for label in range(self._num_classes)
        }

        # 让 Generator 初始化可复现。
        torch.manual_seed(
            self.config.seed
        )

        if backend.device.type == "cuda":
            torch.cuda.manual_seed_all(
                self.config.seed
            )

        forward_model = (
            build_neuroonline_forward(
                runtime_model=runtime_model,
                num_subject_codes=(
                    self.config
                    .num_subject_codes
                ),
                num_attention_heads=(
                    self.config
                    .num_attention_heads
                ),
                dropout=(
                    self.config.dropout
                ),
            )
        )

        # V1 只获取分类头参数。
        #
        # backend 内部会同时冻结 backbone。
        head_parameters = (
            backend
            .get_trainable_parameters(
                scope="head"
            )
        )

        if not head_parameters:
            raise RuntimeError(
                "NeuroOnline backend returned no "
                "trainable head parameters."
            )

        generator_parameters = list(forward_model.generator.parameters())

        # Restore inference mode before applying the explicit scope. Some
        # backends toggle head requires_grad while changing online mode.
        backend.set_online_mode(training=False, train_backbone=False)
        forward_model.generator.eval()

        self._generator_parameters = generator_parameters
        self._head_parameters = list(head_parameters)
        head_parameter_ids = {id(item) for item in self._head_parameters}
        self._backbone_parameters = [
            item
            for item in self._discover_backend_parameters(backend)
            if id(item) not in head_parameter_ids
        ]
        self._apply_parameter_scope()

        trainable_parameters = [
            parameter
            for parameter in [*generator_parameters, *head_parameters]
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("NeuroOnline update scope selected no parameters.")

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=(
                self.config
                .learning_rate
            ),
            weight_decay=(
                self.config
                .weight_decay
            ),
        )

        self._runtime_model = (
            runtime_model
        )

        self._context = context

        self._forward_model = (
            forward_model
        )

        self._trainable_parameters = list(
            trainable_parameters
        )

        self._optimizer = optimizer

        self._validate_parameter_scope()

    def predict_prepared(
        self,
        prepared: PreparedModelInput,
        *,
        return_features: bool = False,
    ) -> ModelOutput:
        """
        使用当前 Generator 状态完成一次预测。

        注意：本方法只预测，不更新参数。
        """

        self._require_initialized()

        output = self.forward_model.predict_prepared(
            prepared,
            return_features=return_features,
        )
        # Model-specific inference mode helpers may toggle the classification
        # head's requires_grad flag. Reassert the configured freeze boundary.
        self._apply_parameter_scope()
        self._validate_parameter_scope()
        return output

    def observe(
        self,
        observation: OnlineObservation,
    ) -> None:
        """
        保存已经完成预测、但尚未收到反馈的输入。

        不使用预测类别作为伪标签。
        """

        self._require_initialized()

        observation_id = (
            observation
            .observation_id
            .strip()
        )

        if not observation_id:
            raise ValueError(
                "observation_id cannot be empty."
            )

        if (
            observation_id
            in self._pending_observations
        ):
            raise ValueError(
                "Duplicate NeuroOnline observation_id: "
                f"{observation_id!r}."
            )

        model_input = (
            observation
            .prepared_input
            .model_input
        )

        _validate_single_sample_input(
            model_input
        )

        buffered_input = (
            _clone_model_input_to_cpu(
                model_input
            )
        )

        self._pending_observations[
            observation_id
        ] = buffered_input

        # 防止长期没有反馈时无限增长。
        while (
            len(self._pending_observations)
            > self.config
            .max_pending_observations
        ):
            self._pending_observations.popitem(
                last=False
            )

    def submit_feedback(
        self,
        feedback: FeedbackEvent,
    ) -> None:
        """
        将真实标签与之前的 observation 配对。

        V1 不支持：
            - reward-only；
            - 伪标签；
            - label=None。
        """

        self._require_initialized()

        if feedback.label is None:
            if feedback.reward is not None:
                raise ValueError(
                    "NeuroOnline V1 does not support "
                    "reward-only feedback. A true "
                    "class label is required."
                )

            raise ValueError(
                "NeuroOnline V1 requires a true "
                "class label."
            )

        if (
            isinstance(
                feedback.label,
                bool,
            )
            or not isinstance(
                feedback.label,
                Integral,
            )
        ):
            raise TypeError(
                "Feedback label must be an integer."
            )

        label = int(
            feedback.label
        )

        backend = (
            self.forward_model.backend
        )

        if not 0 <= label < (
            backend.num_classes
        ):
            raise ValueError(
                "Feedback label is outside the "
                "model class range: "
                f"label={label}, "
                f"num_classes="
                f"{backend.num_classes}."
            )

        observation_id = (
            feedback
            .observation_id
            .strip()
        )

        if (
            observation_id
            not in self._pending_observations
        ):
            raise KeyError(
                "No pending NeuroOnline observation "
                "matches feedback observation_id="
                f"{observation_id!r}."
            )

        model_input = (
            self._pending_observations
            .pop(observation_id)
        )

        sample = _BufferedSample(
            observation_id=observation_id,
            model_input=model_input,
            label=label,
        )

        if self.config.memory_strategy == "recent_fifo":
            self._training_buffer.append(sample)
        else:
            self._balanced_training_buffers[label].append(sample)

        self._feedback_count_total += 1
        self._label_counts_seen[label] += 1
        self._feedback_since_update += 1
        evaluation_ordinal = feedback.metadata.get("trial_ordinal")
        if isinstance(evaluation_ordinal, Integral) and not isinstance(
            evaluation_ordinal, bool
        ):
            self._latest_feedback_evaluation_ordinal = int(evaluation_ordinal)

    def _not_applied_result(
        self,
        reason: str,
        *,
        trigger_satisfied: bool = False,
    ) -> OnlineUpdateResult:
        return OnlineUpdateResult(
            strategy_name=self.name,
            applied=False,
            update_step=(
                self._update_step
            ),
            model_revision=(
                self._model_revision
            ),
            samples_used=0,
            latency_ms=0.0,
            reason=reason,
            metrics={
                "buffered_samples": self.buffered_sample_count,
                "pending_observations": len(
                    self._pending_observations
                ),
                "feedback_since_update": (
                    self
                    ._feedback_since_update
                ),
                **self._telemetry(
                    trigger_reason=reason,
                    trigger_satisfied=trigger_satisfied,
                    update_applied=False,
                ),
            },
        )

    def maybe_update(
        self,
        *,
        runtime_model: RuntimeModel,
    ) -> OnlineUpdateResult:
        """
        满足更新条件时，在最近的有标签样本上训练。

        更新范围由 config.update_scope 显式决定；backbone 始终冻结。

        冻结：
            backbone
        """

        self._require_initialized()

        if (
            runtime_model
            is not self._runtime_model
        ):
            raise ValueError(
                "maybe_update() received a different "
                "RuntimeModel from the one used during "
                "initialize()."
            )

        buffered_count = self.buffered_sample_count
        warmup_count = (
            buffered_count
            if self.config.memory_strategy == "recent_fifo"
            else self._feedback_count_total
        )

        if self._update_step == 0:
            if self.config.update_trigger == "feedback_count" and (
                warmup_count < self.config.warmup_feedback
            ):
                return self._not_applied_result(
                    "waiting for warmup feedback: "
                    f"{warmup_count}/"
                    f"{self.config.warmup_feedback}"
                )
            if (
                self.config.update_trigger == "class_coverage"
                and not self._class_coverage_satisfied()
            ):
                return self._not_applied_result(
                    "waiting for class coverage: "
                    f"seen={self._seen_label_histogram()}, "
                    f"minimum={self.config.min_feedback_per_class}"
                )

        if (
            self._feedback_since_update
            < self.config.update_interval
        ):
            return self._not_applied_result(
                "waiting for update interval: "
                f"{self._feedback_since_update}/"
                f"{self.config.update_interval}"
            )

        if self._optimizer is None:
            raise RuntimeError(
                "NeuroOnline optimizer is missing."
            )

        optimizer = self._optimizer
        backend = self.forward_model.backend
        generator = self.forward_model.generator

        samples = self._select_update_samples()
        if not samples:
            return self._not_applied_result(
                "update trigger satisfied but memory contains no samples",
                trigger_satisfied=True,
            )

        started = time.perf_counter()

        total_weighted_loss = 0.0
        total_examples = 0
        batch_count = 0
        last_gradient_norm = 0.0
        gradient_clipping_count = 0

        train_generator = self.config.update_scope != "head_only"
        train_head = self.config.update_scope != "generator_only"
        backend.set_online_mode(
            training=train_head,
            train_backbone=False,
        )
        self._apply_parameter_scope()
        self._validate_parameter_scope()

        generator.train(mode=train_generator)

        # 每次 update 使用不同但可复现的 shuffle。
        shuffle_generator = (
            torch.Generator()
        )

        shuffle_generator.manual_seed(
            self.config.seed
            + self._update_step
        )

        try:
            for _ in range(
                self.config
                .epochs_per_update
            ):
                permutation = (
                    torch.randperm(
                        len(samples),
                        generator=(
                            shuffle_generator
                        ),
                    )
                    .tolist()
                )

                for start in range(
                    0,
                    len(samples),
                    self.config.batch_size,
                ):
                    selected_indices = (
                        permutation[
                            start:
                            start
                            + self.config
                            .batch_size
                        ]
                    )

                    selected_samples = [
                        samples[index]
                        for index
                        in selected_indices
                    ]

                    batch_model_input = (
                        _stack_model_inputs(
                            [
                                sample
                                .model_input
                                for sample
                                in selected_samples
                            ]
                        )
                    )

                    labels = torch.tensor(
                        [
                            sample.label
                            for sample
                            in selected_samples
                        ],
                        dtype=torch.long,
                        device=backend.device,
                    )

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    result = (
                        self.forward_model
                        .forward_batch(
                            batch_model_input,
                            train_backbone=False,
                        )
                    )

                    loss = F.cross_entropy(
                        result.logits,
                        labels,
                    )

                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            "NeuroOnline update produced "
                            "a non-finite loss."
                        )

                    loss.backward()

                    gradient_norm = (
                        torch.nn.utils
                        .clip_grad_norm_(
                            self
                            ._trainable_parameters,
                            max_norm=(
                                self.config
                                .max_grad_norm
                            ),
                        )
                    )

                    last_gradient_norm = float(
                        gradient_norm
                        .detach()
                        .cpu()
                        .item()
                    )
                    if last_gradient_norm > self.config.max_grad_norm:
                        gradient_clipping_count += 1

                    optimizer.step()

                    current_batch_size = len(
                        selected_samples
                    )

                    total_weighted_loss += (
                        float(
                            loss
                            .detach()
                            .cpu()
                            .item()
                        )
                        * current_batch_size
                    )

                    total_examples += (
                        current_batch_size
                    )

                    batch_count += 1

        finally:
            # 无论成功还是异常，都恢复预测模式。
            backend.set_online_mode(
                training=False,
                train_backbone=False,
            )
            self._apply_parameter_scope()
            generator.eval()

        if total_examples <= 0:
            raise RuntimeError(
                "NeuroOnline update used no samples."
            )

        if self._update_step == 0:
            self._first_update_evaluation_ordinal = (
                self._latest_feedback_evaluation_ordinal
                if self._latest_feedback_evaluation_ordinal is not None
                else self._feedback_count_total
            )
            self._first_update_label_histogram = self._label_histogram(samples)

        self._update_step += 1

        self._model_revision = (
            f"neuroonline-"
            f"{self._update_step}"
        )

        self._feedback_since_update = 0

        latency_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        return OnlineUpdateResult(
            strategy_name=self.name,
            applied=True,
            update_step=(
                self._update_step
            ),
            model_revision=(
                self._model_revision
            ),
            samples_used=len(samples),
            latency_ms=latency_ms,
            reason=None,
            metrics={
                "loss": (
                    total_weighted_loss
                    / total_examples
                ),
                "batches": batch_count,
                "epochs": (
                    self.config
                    .epochs_per_update
                ),
                "buffered_samples": self.buffered_sample_count,
                "last_gradient_norm": (
                    last_gradient_norm
                ),
                "gradient_clipping_count": gradient_clipping_count,
                "gradient_clipping_rate": (
                    gradient_clipping_count / batch_count
                ),
                **self._telemetry(
                    trigger_reason="update_conditions_satisfied",
                    trigger_satisfied=True,
                    update_applied=True,
                    selected_samples=samples,
                ),
                **self.parameter_audit,
                "gate_alpha": float(
                    generator
                    .gate_alpha
                    .detach()
                    .cpu()
                    .item()
                ),
                "gate_beta": float(
                    generator
                    .gate_beta
                    .detach()
                    .cpu()
                    .item()
                ),
            },
        )

    def state_dict(
        self,
    ) -> dict[str, object]:
        """
        保存在线适配状态。

        包括：
            - Generator；
            - head 参数；
            - optimizer；
            - 更新计数；
            - 最近训练 buffer。

        尚未收到反馈的 pending observation 不保存。
        重启后不应再等待旧 session 的反馈。
        """

        self._require_initialized()

        if self._optimizer is None:
            raise RuntimeError(
                "NeuroOnline optimizer is missing."
            )

        generator_state = {
            name: (
                value
                .detach()
                .to(device="cpu")
                .clone()
            )
            for name, value in (
                self.forward_model
                .generator
                .state_dict()
                .items()
            )
        }

        head_parameter_values = [
            parameter
            .detach()
            .to(device="cpu")
            .clone()
            for parameter
            in self._head_parameters
        ]

        buffer_state = [
            {
                "observation_id": (
                    sample.observation_id
                ),
                "model_input": (
                    _clone_model_input_to_cpu(
                        sample.model_input
                    )
                ),
                "label": sample.label,
            }
            for sample
            in self._training_buffer
        ]

        balanced_buffer_state = {
            str(label): [
                {
                    "observation_id": sample.observation_id,
                    "model_input": _clone_model_input_to_cpu(sample.model_input),
                    "label": sample.label,
                }
                for sample in samples
            ]
            for label, samples in self._balanced_training_buffers.items()
        }

        return {
            "strategy_name": self.name,
            "config": asdict(
                self.config
            ),
            "generator_state": (
                generator_state
            ),
            "head_parameter_values": (
                head_parameter_values
            ),
            "optimizer_state": copy.deepcopy(
                self._optimizer.state_dict()
            ),
            "update_step": (
                self._update_step
            ),
            "model_revision": (
                self._model_revision
            ),
            "feedback_since_update": (
                self._feedback_since_update
            ),
            "training_buffer": (
                buffer_state
            ),
            "balanced_training_buffers": balanced_buffer_state,
            "feedback_count_total": self._feedback_count_total,
            "label_counts_seen": self._seen_label_histogram(),
            "latest_feedback_evaluation_ordinal": (
                self._latest_feedback_evaluation_ordinal
            ),
            "first_update_evaluation_ordinal": (
                self._first_update_evaluation_ordinal
            ),
            "first_update_label_histogram": (
                self._first_update_label_histogram
            ),
        }

    def load_state_dict(
        self,
        state: dict[str, object],
    ) -> None:
        """
        恢复状态。

        必须先调用 initialize()，再调用本方法，
        因为需要先创建 Generator、head 和 optimizer。
        """

        self._require_initialized()

        if self._optimizer is None:
            raise RuntimeError(
                "NeuroOnline optimizer is missing."
            )

        strategy_name = state.get(
            "strategy_name"
        )

        if strategy_name != self.name:
            raise ValueError(
                "Cannot load state for strategy "
                f"{strategy_name!r} into "
                f"{self.name!r}."
            )

        saved_config = state.get(
            "config"
        )

        if isinstance(saved_config, dict):
            saved_config = dict(saved_config)
            legacy_defaults = asdict(NeuroOnlineConfig())
            for key in (
                "update_scope",
                "update_trigger",
                "min_feedback_per_class",
                "memory_strategy",
                "balanced_memory_per_class",
            ):
                saved_config.setdefault(key, legacy_defaults[key])

        if saved_config != asdict(
            self.config
        ):
            raise ValueError(
                "Saved NeuroOnline configuration "
                "does not match current configuration."
            )

        generator_state = state.get(
            "generator_state"
        )

        if not isinstance(
            generator_state,
            dict,
        ):
            raise TypeError(
                "generator_state must be a dictionary."
            )

        self.forward_model.generator.load_state_dict(
            generator_state
        )

        head_values = state.get(
            "head_parameter_values"
        )

        if not isinstance(
            head_values,
            list,
        ):
            raise TypeError(
                "head_parameter_values must be a list."
            )

        if len(head_values) != len(
            self._head_parameters
        ):
            raise ValueError(
                "Saved head parameter count does not "
                "match current backend."
            )

        with torch.no_grad():
            for parameter, value in zip(
                self._head_parameters,
                head_values,
                strict=True,
            ):
                if not isinstance(
                    value,
                    torch.Tensor,
                ):
                    raise TypeError(
                        "Saved head parameter must be "
                        "torch.Tensor."
                    )

                if (
                    tuple(parameter.shape)
                    != tuple(value.shape)
                ):
                    raise ValueError(
                        "Saved head parameter shape "
                        "does not match current head."
                    )

                parameter.copy_(
                    value.to(
                        device=(
                            parameter.device
                        ),
                        dtype=(
                            parameter.dtype
                        ),
                    )
                )

        optimizer_state = state.get(
            "optimizer_state"
        )

        if not isinstance(
            optimizer_state,
            dict,
        ):
            raise TypeError(
                "optimizer_state must be a dictionary."
            )

        self._optimizer.load_state_dict(
            optimizer_state
        )

        # 将 optimizer 中的动量等 Tensor
        # 搬到当前 backend 设备。
        for optimizer_item in (
            self._optimizer
            .state
            .values()
        ):
            for key, value in list(
                optimizer_item.items()
            ):
                if isinstance(
                    value,
                    torch.Tensor,
                ):
                    optimizer_item[key] = (
                        value.to(
                            self
                            .forward_model
                            .backend
                            .device
                        )
                    )

        self._update_step = int(
            state.get(
                "update_step",
                0,
            )
        )

        self._model_revision = str(
            state.get(
                "model_revision",
                (
                    f"neuroonline-"
                    f"{self._update_step}"
                ),
            )
        )

        self._feedback_since_update = int(
            state.get(
                "feedback_since_update",
                0,
            )
        )

        self._training_buffer.clear()

        buffer_state = state.get(
            "training_buffer",
            [],
        )

        if not isinstance(
            buffer_state,
            list,
        ):
            raise TypeError(
                "training_buffer must be a list."
            )

        for item in buffer_state:
            if not isinstance(item, dict):
                raise TypeError(
                    "Each training buffer item must "
                    "be a dictionary."
                )

            observation_id = str(
                item["observation_id"]
            )

            label = int(
                item["label"]
            )

            model_input = item[
                "model_input"
            ]

            if not isinstance(
                model_input,
                (torch.Tensor, dict),
            ):
                raise TypeError(
                    "Saved buffer model_input has "
                    "an unsupported type."
                )

            self._training_buffer.append(
                _BufferedSample(
                    observation_id=(
                        observation_id
                    ),
                    model_input=(
                        _clone_model_input_to_cpu(
                            model_input
                        )
                    ),
                    label=label,
                )
            )

        for samples in self._balanced_training_buffers.values():
            samples.clear()
        balanced_buffer_state = state.get("balanced_training_buffers", {})
        if not isinstance(balanced_buffer_state, dict):
            raise TypeError("balanced_training_buffers must be a dictionary.")
        for label_text, items in balanced_buffer_state.items():
            label = int(label_text)
            if label not in self._balanced_training_buffers:
                raise ValueError("Saved balanced-memory label is outside class range.")
            if not isinstance(items, list):
                raise TypeError("Each balanced-memory buffer must be a list.")
            for item in items:
                if not isinstance(item, dict):
                    raise TypeError("Each balanced-memory item must be a dictionary.")
                model_input = item.get("model_input")
                if not isinstance(model_input, (torch.Tensor, dict)):
                    raise TypeError("Saved balanced-memory model_input is invalid.")
                saved_label = int(item["label"])
                if saved_label != label:
                    raise ValueError("Balanced-memory key and sample label differ.")
                self._balanced_training_buffers[label].append(
                    _BufferedSample(
                        observation_id=str(item["observation_id"]),
                        model_input=_clone_model_input_to_cpu(model_input),
                        label=saved_label,
                    )
                )

        restored_samples = [
            *self._training_buffer,
            *[
                sample
                for samples in self._balanced_training_buffers.values()
                for sample in samples
            ],
        ]
        self._feedback_count_total = int(
            state.get("feedback_count_total", len(restored_samples))
        )
        label_counts = state.get("label_counts_seen")
        if label_counts is None:
            label_counts = dict(Counter(sample.label for sample in restored_samples))
        if not isinstance(label_counts, dict):
            raise TypeError("label_counts_seen must be a dictionary.")
        self._label_counts_seen = Counter(
            {int(label): int(count) for label, count in label_counts.items()}
        )
        latest_ordinal = state.get("latest_feedback_evaluation_ordinal")
        self._latest_feedback_evaluation_ordinal = (
            None if latest_ordinal is None else int(latest_ordinal)
        )
        first_ordinal = state.get("first_update_evaluation_ordinal")
        self._first_update_evaluation_ordinal = (
            None if first_ordinal is None else int(first_ordinal)
        )
        first_histogram = state.get("first_update_label_histogram")
        if first_histogram is not None and not isinstance(first_histogram, dict):
            raise TypeError("first_update_label_histogram must be a dictionary.")
        self._first_update_label_histogram = (
            None
            if first_histogram is None
            else {str(label): int(count) for label, count in first_histogram.items()}
        )

        # 不恢复旧 session 中尚未收到反馈的预测。
        self._pending_observations.clear()

        self.forward_model.backend.set_online_mode(
            training=False,
            train_backbone=False,
        )

        self._apply_parameter_scope()
        self._validate_parameter_scope()
        self.forward_model.generator.eval()
