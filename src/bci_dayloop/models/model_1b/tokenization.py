"""1B names for the shared, verified 50M signal-to-token implementation."""

from bci_dayloop.models.model_50m.tokenization import (
    Model50MBatchedInput as Model1BBatchedInput,
    Model50MTokenizedInput as Model1BTokenizedInput,
    Model50MTokenizer,
)


class Model1BTokenizer(Model50MTokenizer):
    """Reuse the verified channel-major variable-window tokenization logic."""

