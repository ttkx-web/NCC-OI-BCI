### 模型构建
```
config = Model50MConfig(
    checkpoint_path="checkpoints/50m/model_deploy.pt",
    classifier_path="checkpoints/50m/test_linear_head.pt",
    device="cpu",
)

model = Model50MAdapter(config)
```

### 预处理
```
preprocessor = Model50MPipelinePreprocessor(
    config,
    channel_names=metadata.channel_names,
    sample_rate=metadata.sample_rate,
    input_unit=metadata.unit,
)
```

### 推理
```
model_input = preprocessor.transform(raw_window)

probabilities = model.predict_proba(
    model_input[None, ...]
)[0]
```

### 输入输出契约
```
raw_window：[C, 2500]，250 Hz 下真实 10 秒窗口
model_input：[64, 1000]，float32
probabilities：[4]，float32
```

### 阶段 0.5 配置
```
replay:
  window_sec: 10.0
  step_sec: 0.5
```

### 已知边界
- `test_linear_head.pt` 未训练；
- 当前结果只用于 Pipeline 验证；
- 10 秒窗口可能跨越原有 4 秒 trial；
- 暂时不能用准确率判断效果；
- 当前滤波为 0.1–75 Hz，但降采样到 100 Hz 后 50 Hz 以上不会保留。
