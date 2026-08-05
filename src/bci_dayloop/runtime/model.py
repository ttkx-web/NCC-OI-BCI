from bci_dayloop.models.base import ModelBackend
from bci_dayloop.preprocessing.base import ModelInputTransform
from bci_dayloop.preprocessing.canonical import SignalCanonicalizer
from bci_dayloop.runtime.types import (
    CanonicalEEGWindow,
    InputContract,
    ModelOutput,
    PreparedModelInput,
    RawEEGWindow,
)


class RuntimeModel:
    def __init__(
        self,
        canonicalizer: SignalCanonicalizer,
        input_transform: ModelInputTransform,
        backend: ModelBackend,
    ) -> None:
        self.canonicalizer = canonicalizer
        self.input_transform = input_transform
        self.backend = backend

    @property
    def input_contract(self) -> InputContract:
        return self.input_transform.input_contract

    def canonicalize(
        self,
        raw_window: RawEEGWindow,
    ) -> CanonicalEEGWindow:
        return self.canonicalizer.transform(raw_window)

    def prepare(
        self,
        raw_window: RawEEGWindow,
    ) -> PreparedModelInput:
        canonical = self.canonicalize(raw_window)
        return self.input_transform.transform(canonical)

    def predict_prepared(
        self,
        prepared: PreparedModelInput,
        return_features: bool = False,
    ) -> ModelOutput:
        return self.backend.predict_tensor(
            model_input=prepared.tensor,
            return_features=return_features,
        )

    def predict(
        self,
        raw_window: RawEEGWindow,
        return_features: bool = False,
    ) -> ModelOutput:
        prepared = self.prepare(raw_window)
        return self.predict_prepared(
            prepared=prepared,
            return_features=return_features,
        )