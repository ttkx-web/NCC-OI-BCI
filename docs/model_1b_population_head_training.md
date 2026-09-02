# 1B 群体线性分类头训练（第一版）

本阶段只训练 frozen 1B backbone 的 final-layer flatten linear head。训练和推理特征
均经 `Model1BBackboneRunner.prepare()` 与 `extract_embeddings()` 生成；产物不是 Runtime
Model Package，也不包含 backbone、optimizer 或预训练 `head.*`。

4 秒 LOSO：

```bash
python scripts/train_1b_population_head.py \
  --split-mode loso --subjects 1 2 3 4 5 6 7 8 9 --target-subject 1 \
  --train-session 0train --validation-session 1test --final-test-session 1test \
  --window-seconds 4 --checkpoint checkpoints/backbones/1b/pretrain_checkpoint_4.pt \
  --device cuda
```

4 秒 within-subject：

```bash
python scripts/train_1b_population_head.py \
  --split-mode within-subject --target-subject 1 \
  --train-session 0train --test-session 1test --validation-ratio 0.2 \
  --window-seconds 4 --checkpoint checkpoints/backbones/1b/pretrain_checkpoint_4.pt \
  --device cuda
```

每个命令只接受一个窗口长度（1、2、3 或 4 秒）。head checkpoint 保存 linear
`head_state_dict`、类别映射、backbone SHA-256、1B architecture、完整 preprocessing
contract、拆分、超参、验证集最优 epoch 与最终测试指标；加载时会拒绝窗口、类别顺序或
backbone hash 不一致的契约。

下一阶段才是 `export_1b_model_package.py`、`load_runtime_package()` 和正式部署接入。
