from pathlib import Path

import numpy as np
import torch

from src.bci_dayloop.models.model_50m.config import (
    Model50MConfig,
)
from src.bci_dayloop.models.model_50m.tokenization import (
    Model50MTokenizer,
)


config = Model50MConfig(
    checkpoint_path=Path("checkpoints/50m/model.pt"),
    device="cpu",
)

# 模拟预处理完成后的 64 通道、10 秒 EEG。
signal = np.random.randn(
    config.n_channels,
    config.target_num_points,
).astype(np.float32)

channel_valid_mask = np.ones(
    config.n_channels,
    dtype=np.float32,
)

# 模拟最后 10 个通道缺失。
channel_valid_mask[-10:] = 0.0
signal[-10:] = 0.0

tokenizer = Model50MTokenizer(config)

tokens = tokenizer.tokenize(
    signal=signal,
    channel_valid_mask=channel_valid_mask,
)

print("token_inputs:", tokens.token_inputs.shape)
print(
    "token_channel_indices:",
    tokens.token_channel_indices.shape,
)
print(
    "token_time_indices:",
    tokens.token_time_indices.shape,
)
print(
    "token_valid_mask:",
    tokens.token_valid_mask.shape,
)
print("valid tokens:", tokens.valid_token_count)

batch = tokens.as_batch(device="cpu")

print("batched token_inputs:", batch.token_inputs.shape)
print(
    "batched token_channel_indices:",
    batch.token_channel_indices.shape,
)
print(
    "batched token_time_indices:",
    batch.token_time_indices.shape,
)
print(

    "Replay window:",

    replay_config["window_sec"],

)



print(

    "Decoder window samples:",

    decoder.window_samples,

)

assert tokens.token_inputs.shape == (640, 100)
assert tokens.token_inputs.dtype == torch.float32

assert tokens.token_channel_indices.shape == (640,)
assert tokens.token_channel_indices.dtype == torch.int64

assert tokens.token_time_indices.shape == (640,)
assert tokens.token_time_indices.dtype == torch.int64

assert tokens.token_valid_mask.shape == (640,)
assert tokens.token_valid_mask.dtype == torch.float32

assert batch.token_inputs.shape == (1, 640, 100)
assert batch.token_channel_indices.shape == (1, 640)
assert batch.token_time_indices.shape == (1, 640)