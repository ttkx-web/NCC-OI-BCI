from pathlib import Path

import numpy as np

import sys
from pathlib import Path

# NCC-OI-BCI 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 把 src 加入 Python 模块搜索路径
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.models.model_50m.preprocessing import (
    Model50MPreprocessor,
)


config = Model50MConfig(
    checkpoint_path=Path("checkpoints/50m/model.pt"),
    device="cpu",
    filter_low_hz=0.1,
    filter_high_hz=75.0,
    reference_mode="none",
)

preprocessor = Model50MPreprocessor(config)

# 示例：22 通道、250 Hz、10 秒，原始单位为 V。
raw_eeg = np.random.randn(22, 2500).astype(np.float32) * 20e-6

channel_names = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

result = preprocessor(
    signal=raw_eeg,
    channel_names=channel_names,
    original_sample_rate=250.0,
    input_unit="V",
)

print("signal shape:", result.signal.shape)
print("signal dtype:", result.signal.dtype)
print("mask shape:", result.channel_valid_mask.shape)
print("mapped channels:", result.mapped_channel_count)
print("missing channels:", result.missing_channel_count)
print("notes:", result.notes)

assert result.signal.shape == (64, 1000)
assert result.signal.dtype == np.float32
assert result.channel_valid_mask.shape == (64,)
assert np.isfinite(result.signal).all()