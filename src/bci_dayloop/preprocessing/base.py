from abc import ABC, abstractmethod

from bci_dayloop.runtime.types import (
    CanonicalEEGWindow,
    InputContract,
    PreparedModelInput,
)


class ModelInputTransform(ABC):
    @property
    @abstractmethod
    def input_contract(self) -> InputContract:
        """返回该模型要求的输入规范。"""

    @abstractmethod
    def transform(
        self,
        window: CanonicalEEGWindow,
    ) -> PreparedModelInput:
        """把规范化 EEG 转换为模型 Tensor。"""