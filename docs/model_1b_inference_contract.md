# 1B 模型推理契约审计

审计日期：2026-09-02。范围仅为推理契约核验；没有实现 1B adapter、导出
Runtime Model Package 或训练分类头。

正式审计对象：

```text
/Volumes/UBUNTU-SERV/NCC-OI-BCI/checkpoints/backbones/1b/pretrain_checkpoint_4.pt
```

上游源码审计对象（`main`）：

```text
https://github.com/AbabababaSmart/EEG_pretrain
```

## 后续确认（实施基线）

本文件初版记录了 checkpoint 未序列化通道表、以及公开源码两处通道模板不一致的
审计事实。后续已由项目方确认：**正式 1B 训练实际使用的标准通道表、输入单位、
滤波、参考方式、重采样与 Z-score 均与当前 50M Runtime 一致。** 因此本文中把
`A1,A2`/`F9,F10` 标为「待确认」或 blocker 的历史审计结论，已不再是 1B latency-only
接入的部署 blocker；实现以 `Model50MConfig` 的标准通道与已验证 preprocessing 为
准。checkpoint 仍未自行携带该份外部确认的通道/单位元数据。

## 当前 1B backbone 开发状态（2026-09-02）

本仓库现有的可复用基础执行链路是
`RawEEGWindow → Model1BPreparedInput → final encoder embedding`，公开入口为
`Model1BBackboneRunner.prepare(raw_window)` 与
`Model1BBackboneRunner.extract_embeddings(prepared)`：

| 字段 | 当前实现值 | 证据文件 | 结论 |
|---|---|---|---|
| prepared token 输入 | 1/4/10 秒分别为 `[1,64,100]` / `[1,256,100]` / `[1,640,100]`；token 为 `float32`，channel/time index 为 `int64` | `src/bci_dayloop/models/model_1b/runner.py`; `config.py` | 只接受 1–10 个整秒 patch；超过 10 秒、短于 1 秒和非整秒立即报错 |
| embedding 输出 | `[B,64*num_time_patches,2048]`，取 final encoder layer（index 19） | `runner.py:Model1BBackboneRunner.extract_embeddings`; `backbone.py:Model1BBackbone.extract_embeddings` | 输出仅为特征，不是 logits 或类别 |
| 原始窗口预处理 | 使用内部已验证的 50M 通道映射、滤波、参考、重采样、Z-score 和 channel-major tokenization | `model_1b/preprocessing.py`; `model_1b/tokenization.py` | 不依赖外部 `EEG_pretrain` 仓库或 `sys.path` 注入 |
| checkpoint 加载 | 只载入精确的 `tokenizer.*`、`channel_embed.*`、`time_embed.*`、`encoder.*`；仅忽略并报告 `head.*` | `model_1b/backbone.py:load_backbone_checkpoint` | `head.*` 是 TimeFreqTokenHead 预训练重建头；不存在分类头 |
| 分类部署 | 无分类 logits、概率或标签输出 | `model_1b/runner.py`; formal checkpoint `model_state_dict` | **不能直接用于分类部署**；下一阶段须训练分类头并导出正式 Runtime Model Package |

## 审计方法和 checkpoint 格式

checkpoint 是 PyTorch ZIP 序列化文件（内含
`pretrain_checkpoint_4/data.pkl` 与 tensor storages），文件大小为 3.8 GB。实际
读取其 pickle 元数据和所有 tensor shape（不读取/复制 tensor payload）得到以下
顶层结构；参数总量由 `model_state_dict` 的所有 tensor `numel` 求和。

| 字段 | 1B 实际值 | 证据文件或 checkpoint key | 与 50M 是否一致 | 结论 |
|---|---|---|---|---|
| checkpoint 顶层 keys | `model_state_dict`, `config`, `shape_params`, `epoch`, `train_loss`, `val_loss` | `pretrain_checkpoint_4.pt:data.pkl` | 50M package 不是这个预训练 checkpoint 格式 | 可按上游训练保存逻辑解析；不可把它当 50M package |
| state dict 来源 | `model_state_dict = model_to_save.state_dict()`；DDP 时 `model.module` | 上游 `largeScale_pretrain_main.py:586-598`; key `model_state_dict` | 不同 | key 没有 `module.` 前缀 |
| 保存 config | 可反序列化的上游 `Config` 对象，含 `data` / `mask` / `model` / `train` | checkpoint key `config` | 50M 以 `Model50MConfig`/package YAML 表示 | 已实际读取；不可用源码默认值替代 |
| 保存 shape_params | `patch_len=100`, `patch_stride=100`, `total_points=1000`, `num_time_patches=10`, `num_tokens=640`, `frames_per_patch=5`, `n_bands=5` | checkpoint key `shape_params` | 仅 50M 10 s 配置的 token shape 相同 | 1B 的正式 token shape 已确认 |
| epoch / loss | `epoch=4`, `train_loss=0.29317352175712585`, `val_loss=0.27948710322380066` | checkpoint top-level keys | 不适用 | 预训练 epoch checkpoint，不是下游部署头 |
| 参数总量 | **1,011,775,513**（全部为 `FloatStorage`） | `model_state_dict` 250 entries 的 shape/numel 实测 | 50M config 为 512-dim/12 层，量级不同 | 1B，不可套用 50M backbone loader |

## 已实际核验的 backbone 与权重完整性

| 字段 | 1B 实际值 | 证据文件或 checkpoint key | 与 50M 是否一致 | 结论 |
|---|---|---|---|---|
| `d_model` | `2048` | `config.model.d_model`; `tokenizer.proj.0.weight=(2048,100)` | 否，50M=`512` | 必须新建 1B 专用构造/加载路径 |
| `n_heads` | `16` | `config.model.n_heads` | 否，50M=`8` | 必须调整 |
| `depth` | `20` | `config.model.depth`; layers `0..19` 共 20 组 | 否，50M=`12` | 必须调整 |
| `mlp_ratio` | `4.0`（FFN 为 `8192=4*2048`） | `config.model.mlp_ratio`; `encoder.encoder.layers.0.linear1.weight=(8192,2048)` | 是 | 可复用该概念，不能复用权重 loader |
| `dropout` | `0.1` | `config.model.dropout` | 是 | eval 推理时 dropout 关闭 |
| tokenizer | `tokenizer.proj.0.{weight,bias}`、`tokenizer.proj.1.{weight,bias}` 均存在 | 4 个 `tokenizer.*` keys | 结构概念相同、维度不同 | 齐全 |
| channel embedding | `channel_embed.weight=(64,2048)` | key `channel_embed.weight` | 64 通道相同，维度不同 | 齐全；**不证明通道名称顺序** |
| time embedding | `time_embed.weight=(10,2048)` | key `time_embed.weight` | 50M 10-s backbone 的 10 位置相同 | 齐全；只接受 time index `0..9` |
| encoder | `encoder.encoder.layers.0..19`，每层 attention/FFN/norm 共 12 keys | 240 个 `encoder.*` keys | 否，50M 12 层 | 齐全 |
| 预训练重建 head | `head.head.0.{weight,bias}`，`head.head.2.{weight,bias}`；末层 `(25,2048)` | 4 个 `head.*` keys；`n_bands=5`, `frames_per_patch=5` | 50M Runtime 不使用此重建 head | 齐全，但它是 `TimeFreqTokenHead`，不是分类头 |
| 下游分类头 | **不存在**：250 keys 只覆盖上述五个前缀，未见 `classifier`、`linear_probe`、`fc` 或任何类别数输出层 | 完整 `model_state_dict` key 枚举；上游 `model.py:TimeFreqTokenHead` | 否，50M package 另有 trained linear probe | **不能直接用于分类部署** |

## 1B 完整输入契约

### 原始信号与预处理

| 字段 | 1B 实际值 | 证据文件或 checkpoint key | 与 50M 是否一致 | 结论 |
|---|---|---|---|---|
| 原始输入 shape | 训练 builder 接收 `[C_in,T]`；其固定 raw Lance 假设为 `ORIG_SFREQ=200.0`，随后得到 `[64,1000]` | 上游 `build_processed_lance_from_raw.py:234-255,499-510` | 50M Runtime 外部接口为 `[C,T]`，Stage 2B 现场输入为 `[59,4000]@1000 Hz` | 接口维数形式相同；时长/采样率链路不同 |
| 原始物理单位 | **未由正式 checkpoint 或 builder 声明，待确认。** builder 仅执行 `signal = signal * 200.0`（`DENORM_CLIP_DIVIDE_200=True`）；虽读取 `config.data.convert_v_to_uv=True`，但此变量在 builder 未用于单位换算 | `build_processed_lance_from_raw.py:30-35,53,254-255`; checkpoint `config.data.convert_v_to_uv=True` | 否，50M 明确把 `V/mV/uV` 转为 `uV`，Stage 2B 要求 `uV` | 不得把 50M 的 `uV` 输入直接声明为 1B 已验证单位；需追溯 raw Lance 生成链路/单位元数据 |
| 标准 64 通道（训练 builder 暂定顺序） | `Fp1,Fpz,Fp2,AF7,AF3,AFz,AF4,AF8,F7,F5,F3,F1,Fz,F2,F4,F6,F8,FT7,FC5,FC3,FC1,FCz,FC2,FC4,FC6,FT8,T7,C5,C3,C1,Cz,C2,C4,C6,T8,TP7,CP5,CP3,CP1,CPz,CP2,CP4,CP6,TP8,P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,PO3,POz,PO4,PO8,O1,Oz,O2,Iz,A1,A2` | `build_processed_lance_from_raw.py:149-162` | 前 62 位一致；末两位不同：50M=`F9,F10` | **待确认，不能复用 50M 映射** |
| 上游通道定义冲突 | `channel_config.py` 定义末两位为 `F9,F10`；训练 builder 自己硬编码为 `A1,A2`，且没有 import `channel_config.STANDARD_64_CHANNELS` | 上游 `channel_config.py:STANDARD_64_CHANNELS`; builder `:149-162` | `src/bci_dayloop/models/model_50m/config.py:STANDARD_64_CHANNELS` 采用 `F9,F10` | 这是 1B Runtime 接入 blocker；checkpoint `config` 仅有 `n_channels=64`，无法仲裁 |
| 通道别名 | builder：去除 `EEG ` 与 `-REF/-LE/-A1/-A2/-M1/-M2` 后缀；`T3→T7,T4→T8,T5→P7,T6→P8`，`M1→A1,M2→A2`，并规范大小写 | `build_processed_lance_from_raw.py:175-229` | 50M 只有同名和部分 10-20/大小写别名，未含 `M1/M2→A1/A2` | 必须调整，并先解决 `A1/A2` 争议 |
| 未知通道 | 不在 template 的名称直接跳过 | builder `align_to_standard_channels():299-307` | 50M 同样忽略 unknown 并记录 diagnostics | 概念一致；部署应显式记录/按 Runtime policy fail-closed |
| 重复通道 | 对映射到同一目标的输入信号求均值；不报错 | builder `align_to_standard_channels():299-319` | 50M 也平均重复映射通道 | 可复用算法；若遵守 Stage 2B policy，重复应先 fail-closed |
| 缺失通道 | 全零填充，`channel_valid_mask=0.0` | builder `align_to_standard_channels():312-319`; 上游 `channel_config.py` 文件说明 | 50M 亦全零/valid mask | 概念一致；1B `forward()` 不接收该 mask，缺失位置仍以零 patch 与其 channel id 进入 encoder |
| 平均参考 | 无此处理（builder 仅 map → resample → crop/pad → Z-score） | builder `preprocess_one_raw_sample():505-510` | 50M `reference_mode=none` 的 package 一致；50M 代码可选 average | 1B 必须为 `none`；不能启用 50M 的可选 average |
| 滤波 | 无 band-pass/notch/filter 调用 | builder imports/processing path `:13,505-517`（唯一 `scipy.signal` 特征计算为 target STFT） | 否，50M packages 开启 0.1–75 Hz 4 阶 band-pass | **必须调整：1B adapter 不得默认复用 50M filter** |
| 目标采样率 | `100.0 Hz` | checkpoint `config.data.target_sfreq=100.0`；`shape_params.total_points=1000` | 是 | 可复用重采样目标 |
| 窗口长度 / stride | `clip_seconds=10.0`，`clip_stride_seconds=10.0` | checkpoint `config.data.*` | 50M Stage05 10-s package 一致；Stage2B 4-s package 不一致 | 1B 必须 10.0 s；不能接收 4.0-s Runtime prepared input |
| patch | `patch_seconds=1.0`，`patch_len=100`，`patch_stride_seconds=1.0`，`patch_stride=100`，10 个时间 patch | checkpoint `config.data.*`, `shape_params.*` | 50M 10-s/4-s 都是 1s/100 points；4-s 只有 4 patches | 10-s 50M token 切法可复用；4-s 不可 |
| 每窗归一化 | 每个有效通道沿整个 1000 samples 作 Z-score：`(x-mean)/(std+1e-8)`；缺失通道保持 0 | builder `standardize_signal_by_channel():355-369,510` | 50M 亦有效通道的整窗、沿时间维 Z-score，`eps=1e-8` | 在关闭 50M filter 且通道映射获确认的前提下可复用 |
| token 排列 | `[C,N,L]` reshape 至 `[C*N,L]`，channel-major：每 channel 的 time `0..9` 连续 | builder `:519-536` | 50M tokenizer 相同排列 | 10-s 形状可直接复用 |
| `token_inputs` | `torch.float32`, `[B,640,100]`（无 batch 时 `[640,100]`） | Lance builder dtype/shape `:532,552-555`; checkpoint `shape_params` | 50M 10-s 相同；4-s 是 `[B,256,100]` | 仅 10-s tokenization 可复用 |
| `token_channel_indices` | `torch.int64`, `[B,640]`；`[0]*10,[1]*10,...,[63]*10` | builder `:534,574` | 50M 10-s 相同 | 可直接复用，仅在正式通道顺序确认后 |
| `token_time_indices` | `torch.int64`, `[B,640]`；`0..9` 重复 64 次 | builder `:535,575`; `time_embed.weight=(10,2048)` | 50M 10-s 相同 | 可直接复用；4-s `0..3` 不满足本次完整契约 |
| 非模型 mask 元数据 | `token_valid_mask`: `float32 [B,640]`；`channel_valid_mask`: `float32 [B,64]` | builder `:536-544`; training entry读入这些字段 | 50M 会保留并在 pooling 使用 mask | 1B `EEGPretrainModel.forward()` 不收这两个参数；可保留供质量门禁/未来下游 head，但不能假装 encoder 使用它们 |

`token_inputs` 之前的实际训练路径为：`[C_in,T] → 通道对齐/缺失补零 →
resample(100 Hz) → crop/pad(1000) → 有效通道 Z-score → [64,10,100] →
[640,100]`。builder 对长窗口取前 1000 点、短窗口在尾部补零；生产接入应要求
真实 10.0-s 窗口而不是依赖该静默补齐行为。

## 输出与下游 embedding 契约

| 字段 | 1B 实际值 | 证据文件或 checkpoint key | 与 50M 是否一致 | 结论 |
|---|---|---|---|---|
| 当前 `EEGPretrainModel.forward()` 输出 | `pred: [B,S,F,K] = [B,640,5,5]` | 上游 `model.py:TimeFreqTokenHead/EEGPretrainModel.forward`; checkpoint `shape_params` 与 `head.head.2.weight=(25,2048)` | 否，50M Runtime backbone 输出 embeddings | 语义是每个通道-时间 token 的 5 个频带、每 patch 5 帧的标准化功率轨迹预测 |
| forward 输入 | 仅 `token_inputs`, `token_channel_indices`, `token_time_indices` | 上游 `model.py:EEGPretrainModel.forward` | 50M backbone 同三项、其下游另用 valid mask | 不接受 classification labels/mask 参数 |
| 可供分类的 embedding | 取 `x = encoder(tokenizer(token_inputs)+channel_embed+time_embed)`、即调用 `head` **之前**的最终 encoder 输出；shape `[B,640,2048]` | 上游 `model.py:137-184` | 概念类似 50M `extract_token_embeddings`，但 1B 没有该公开方法 | 接入时需要专用只读 wrapper/adapter 暴露该张量；再由明确的 pooling/flatten + 已训练 head 定义分类 |
| 中间层 embedding | 当前上游 `EEGTransformerEncoder.forward()` 只返回最终层；无 `return_layer_idx`/hook 接口 | 上游 `model.py:29-53` | 否，50M 有 `output_layer_idx` | 当前合同只指定最终第 20 层输出；若要中间层，须另行定义 adapter 合同 |
| 可用分类 logits | **没有。**唯一 head 输出 25=5 bands×5 frames，不是 classes；checkpoint 也没有分类 key | `head.head.2.weight=(25,2048)`；完整 key 列表 | 否，50M package 的 `classifier_type=trained_linear_probe` | **不能直接用于分类部署** |

## 与当前 NCC-OI-BCI 50M Runtime 的逐项结论

当前仓库同时存在 10-s Stage05 package 和 4-s Stage2B approved package：

* `model_packages/stage05/bnci2014_001/subject_01/population/10s_flatten/v1/model.yaml`
  为 10.0 s / 100 Hz / 640 tokens；
* `docs/stage2b/realtime_runtime_input_contract.md` 规定已批准 realtime
  prepared contract 为 `[1,64,400]` / 4.0 s / 57 valid channels，且正式 target
  template 必须含 `F9,F10`。

| 字段 | 1B 实际值 | 证据文件或 checkpoint key | 与 50M 是否一致 | 结论 |
|---|---|---|---|---|
| `Model50MConfig` backbone | `2048/16/20/4.0/0.1` | 1B checkpoint `config.model`; `src/bci_dayloop/models/model_50m/config.py:110-116` 为 `512/8/12/4.0/0.1` | 仅 ratio/dropout 一致 | 必须调整，不能复用 `Model50MBackbone`/loader |
| 50M preprocessing 输入单位 | 1B 未证实单位；builder 有 `*200` 数值反归一化 | 1B builder `:30-35,254-255`; 50M `preprocessing.py:535-543` | 不一致/待确认 | 必须调整；先完成 1B 单位溯源 |
| 通道 template | 1B builder=`…Iz,A1,A2`；源码另一文件/50M=`…Iz,F9,F10` | 1B builder `:149-162`; 1B `channel_config.py`; 50M `config.py:STANDARD_64_CHANNELS` | 否 | **待确认且 blocker**；Stage2B 59→64 mapping 不能直接复用 |
| 缺失/重复/unknown | 1B：zero-fill + mask、duplicate average、unknown skip | 1B builder `:286-319` | 50M 同一基本算法；Stage2B policy 禁止额外修复 | 算法可直接复用；Runtime policy须为 1B 单独重新审批 |
| 平均参考 | 1B none | builder `:505-510` | 50M package `reference_mode: none` | 可直接复用 `none`，不得切换到 `average` |
| 滤波 | 1B none | 1B builder `:505-517` | 50M package `filter_enabled: true`, `0.1–75 Hz`, order 4 | 必须调整：禁用 50M filter 才接近训练链路 |
| Z-score | 有效通道全窗时间维，`eps=1e-8` | 1B builder `:355-369` | 50M `preprocessing.py:616-625` 同方式 | 可直接复用（在前述 filter/channel 条件满足后） |
| 采样率 | 100 Hz | 1B checkpoint `config.data.target_sfreq` | 50M config/package=100 Hz | 可直接复用 |
| 窗口 | 10.0 s / 1000 points / 640 tokens | 1B checkpoint `shape_params` | 50M Stage05 一致；Stage2B current prepared input 4.0 s / 400 points / 256 tokens 不一致 | 仅 10-s 50M 链路可复用 tokenization；当前 4-s Runtime package 必须调整 |
| patch/tokenization | 1.0 s / 100 points / stride 100，channel-major indices | 1B checkpoint `shape_params`; builder `:463-536` | 50M 10-s 完全一致 | 在 10-s、confirmed channel mapping、no-filter 条件下可直接复用 |
| Runtime package 输入约束 | 1B 需要 `[B,640,100]` 三个模型 tensor；当前 Stage2B package输出 `[1,64,400]` + mask | 1B `model.py`; `docs/stage2b/realtime_runtime_input_contract.md:prepared 50M contract` | 否 | 必须有 1B 专用 preprocess/tokenize contract，不能把 4-s prepared input直连 1B |
| 分类输出 | 1B 无 logits/head | 1B state keys/head shape | 50M packages 装载分类 probe | 必须先训练并封装经验证的 1B 下游 head，才可分类部署 |

## 接入门槛（最小事项）

在创建任何 Runtime Model Package 前，必须先完成以下最小闭环：

1. 提供训练 `pretrain_checkpoint_4.pt` 实际使用的 processed Lance schema/sample
   metadata，或对应不可变 commit/run manifest，以确认末两通道究竟是 `A1,A2` 还是
   `F9,F10`，并确认 raw Lance 的物理单位与 `signal * 200` 的含义。
2. 定义并测试专用 1B preprocessing adapter：拒绝不满足真实 10.0-s 窗口的输入，按
   已确认 64 通道映射处理，禁用 50M 的 0.1–75 Hz filter，采用 none reference、100 Hz
   重采样、有效通道全窗 Z-score 和 `[B,640,100]` tokenization。
3. 定义一个仅暴露最终 `encoder` `[B,640,2048]` 的 1B inference wrapper，并验证
   tokenizer/channel/time indices、dtype 和 checkpoint 所有前缀严格加载；不要把
   `TimeFreqTokenHead` 当分类头。
4. 使用明确 pooling/flatten 规则训练、验证、保存并版本化真正的下游分类头与标签
   映射；在此之前该 checkpoint **不能直接用于分类部署**。
