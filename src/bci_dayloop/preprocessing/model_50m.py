from base import ModelInputTransform
from base import InputContract
from bci_dayloop.runtime.types import CanonicalEEGWindow
from bci_dayloop.runtime.types import PreparedModelInput

class Model50MInputTransform(ModelInputTransform):
    def __init__(self, config: dict) -> None:
        self.config = config
        self._contract = InputContract(
            channel_names=tuple(config["channel_names"]),
            sample_rate=float(config["sample_rate"]),
            window_sec=float(config["window_sec"]),
            num_samples=int(config["num_samples"]),
            input_unit=config["input_unit"],
            tensor_layout=config.get("tensor_layout", "BCT"),
            strict_window_duration=config.get(
                "strict_window_duration",
                True,
            ),
        )

    @property
    def input_contract(self) -> InputContract:
        return self._contract

    def transform(
        self,
        window: CanonicalEEGWindow,
    ) -> PreparedModelInput:
        trace = list(window.processing_history)
        data = window.data

        # 下面不要凭空重新设计。
        # 把当前 50MModelAdapter 中已经验证过的预处理顺序原样迁移过来。

        data = self._resample_if_needed(data, window.sample_rate)
        trace.append(
            f"resample:{window.sample_rate}->{self._contract.sample_rate}"
        )

        data = self._select_and_reorder_channels(
            data=data,
            source_names=window.channel_names,
            target_names=list(self._contract.channel_names),
        )
        trace.append("select_and_reorder_50m_channels")

        data = self._validate_or_crop_window(data)
        trace.append(
            f"validate_window:{self._contract.num_samples}_samples"
        )

        data = self._apply_training_normalization(data)
        trace.append("apply_50m_training_normalization")

        tensor = self._to_tensor(data)
        trace.append(
            f"to_tensor:{self._contract.tensor_layout}"
        )

        return PreparedModelInput(
            tensor=tensor,
            canonical_window=window,
            preprocessing_trace=trace,
        )