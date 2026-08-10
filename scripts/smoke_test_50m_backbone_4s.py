from pathlib import Path

import numpy as np

from bci_dayloop.models.model_50m.backbone import (
    Model50MBackbone,
)
from bci_dayloop.models.model_50m.classifier import (
    Model50MClassifier,
)
from bci_dayloop.models.model_50m.config import (
    Model50MConfig,
)
from bci_dayloop.models.model_50m.preprocessing import (
    Model50MPreprocessor,
)
from bci_dayloop.models.model_50m.tokenization import (
    Model50MTokenizer,
    stack_model50m_tokens,
)

from _bootstrap import ROOT


config = Model50MConfig(
    checkpoint_path=(
            ROOT / "checkpoints/backbones/50m/model_deploy.pt"
    ),
    device="cpu",
    window_seconds=4.0,
    model_n_time_patches=10,
    target_sample_rate=100.0,
    patch_seconds=1.0,
    patch_stride_seconds=1.0,
    aggregation="flatten",
)

print("target_num_points:", config.target_num_points)
print("input time patches:", config.num_time_patches)
print("model time patches:", config.model_n_time_patches)
print("num_tokens:", config.num_tokens)
print("feature_dim:", config.classifier_input_dim)

assert config.target_num_points == 400
assert config.num_time_patches == 4
assert config.model_n_time_patches == 10
assert config.num_tokens == 256
assert config.classifier_input_dim == 131072

backbone = Model50MBackbone(
    config=config,
    load_checkpoint=True,
    freeze=True,
)

classifier = Model50MClassifier(
    config=config,
    backbone=backbone,
)

fake_signal = np.random.randn(
    64,
    400,
).astype(np.float32)

tokenizer = Model50MTokenizer(config)

tokenized = tokenizer.tokenize(
    signal=fake_signal,
    channel_valid_mask=np.ones(
        64,
        dtype=np.float32,
    ),
)

batch = stack_model50m_tokens([tokenized])

features = classifier.extract_features(batch)

print("tokens:", tokenized.token_inputs.shape)
print("features:", features.shape)

assert tuple(tokenized.token_inputs.shape) == (
    256,
    100,
)
assert tuple(features.shape) == (
    1,
    131072,
)

print("4-second backbone smoke test passed.")