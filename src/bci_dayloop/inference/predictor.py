from __future__ import annotations

from typing import (
    Protocol,
    runtime_checkable,
)

from bci_dayloop.runtime.types import (
    ModelOutput,
    PreparedModelInput,
)


@runtime_checkable
class PreparedPredictor(Protocol):
    """
    接收已经完成预处理的模型输入，
    返回统一 ModelOutput 的预测接口。

    RuntimeModel 和 NeuroOnlineStrategy
    都可以实现这个接口。
    """

    def predict_prepared(
        self,
        prepared: PreparedModelInput,
        *,
        return_features: bool = False,
    ) -> ModelOutput:
        ...