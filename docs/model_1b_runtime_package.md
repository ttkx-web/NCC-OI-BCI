# 1B Runtime Model Package

一个 frozen 1B flatten linear head 对应一个固定窗口长度的 Runtime Model Package。
预训练 backbone 始终保留 10 个 time-position embeddings，但 Package 的实际窗口可以是
1–10 个整秒 patch；`window_sec`、`num_time_patches`、`token_count` 与
`classifier_input_dim` 全部从 head checkpoint metadata 导出，不能由 Runtime 改写。

当前 4 秒群体 head 的导出示例：

```bash
python scripts/export_1b_model_package.py \
  --backbone-checkpoint checkpoints/backbones/1b/pretrain_checkpoint_4.pt \
  --head-checkpoint /path/to/1b_4s_population_head.pt \
  --output-dir model_packages/stage1_1b/bnci2014_001/subject_01/population/4s_flatten/v1 \
  --device cuda
```

该 Package 会记录 `window_sec=4`、`num_time_patches=4`、`token_count=256` 和
`classifier_input_dim=524288`。以后训练出 1/2/3/10 秒 head 时，只需替换
`--head-checkpoint` 和对应输出目录；无需修改 exporter、loader 或 Runtime。

离线 replay 使用既有脚本：

```bash
python scripts/replay_offline.py --model-package /path/to/1b_package
```

Runtime 会拒绝不匹配 Package `window_sec` 的输入，且不会补零、裁剪或以另一长度 head
代替。下一阶段才涉及更高级的部署服务与个人化策略。
