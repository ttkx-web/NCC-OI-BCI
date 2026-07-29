# BCI DayLoop

An independent Windows-first BCI infrastructure project for a one-day path from
BNCI2014_001 Subject 1 to a frozen LaBraM Base encoder, cached embeddings, a
linear motor-imagery head, pseudo-realtime replay, and a Streamlit dashboard.

The project does not import from or modify `rest_tune_base` or `oi-armi`. Its
interfaces follow their useful adapter/factory/replay patterns, while the
PyTorch-only LaBraM backbone follows the official `labram_base_patch200_200`
state-dict layout and channel-position indexing.

## 50M Model Adapter

阶段 0.5 的 50M 模型适配说明见：

- [50M Model Adapter README](docs/README_50M_MODEL_ADAPTER.md)

## Pipeline

1. A separate data-preparation environment uses MOABB to download BNCI2014_001 Subject 1.
2. The two sessions are written to one HDF5 file as `float32 [N,C,T]` in volts.
3. One shared preprocessor removes non-EEG channels, applies 0.1–75 Hz bandpass
   and 50 Hz notch filters, resamples to 200 Hz, converts to µV, performs
   per-window/per-channel Z-score, and reshapes to `[B,C,A,200]`.
4. The first session (`0train`) is stratified into train/validation sets.
5. LaBraM Base is frozen. Embeddings are cached under the run directory.
6. Only `torch.nn.Linear` is trained. The second session (`1test`) is untouched
   until final evaluation and replay.
7. A standard package is written to `runs/day1_bnci_s01/model_package/`.

Commands map as follows:

| Class | Command |
|---|---|
| `left_hand` | `LEFT` |
| `right_hand` | `RIGHT` |
| `feet` | `FORWARD` |
| `tongue` | `STOP` |

Any prediction below the configured confidence threshold is forced to `STOP`.

## Windows environment

The project intentionally uses two environments:

- `bci-dayloop`: main training, replay, model loading, and Streamlit runtime.
- `bci-dayloop-data`: optional data-download environment containing MOABB.

The main runtime does not depend on MOABB. It reads the already generated
`data/processed/bnci2014_001_s01.h5` directly.

### Main training/replay environment

From PowerShell or Anaconda Prompt:

```powershell
Set-Location E:\code\BCI_DayLoop
conda env create -f environment.yml
conda activate bci-dayloop
python -m pip install -e .
```

The pinned main runtime is Python 3.11, NumPy 1.26.4, PyTorch 2.0.1,
TorchVision 0.15.2, and TorchAudio 2.0.2. Alternatively, run `setup_env.bat`
and then activate the environment.

### Optional data-preparation environment

Use this only when you need to download or regenerate the HDF5 file. It is
separate from the main runtime so training and replay do not need MOABB:

```powershell
Set-Location E:\code\BCI_DayLoop
conda create -n bci-dayloop-data python=3.11 -y
conda activate bci-dayloop-data
python -m pip install -r requirements-data.txt
$env:MNE_DATASETS_BNCI_PATH = "E:\code\BCI_DayLoop\data\moabb_cache"
New-Item -ItemType Directory -Force data\moabb_cache | Out-Null
python scripts\prepare_bnci2014_001.py --config configs\day1_bnci_s01.yaml
conda deactivate
```

After the HDF5 exists, switch back to `bci-dayloop` for all training, replay,
and web commands.

## LaBraM checkpoint

Download the official LaBraM Base fine-tuning checkpoint from the
[official LaBraM repository](https://github.com/935963004/LaBraM) and place it at:

```text
E:\code\BCI_DayLoop\checkpoints\labram-base.pth
```

The normal pipeline fails early with an explicit path and remediation message
if this file is missing. Random initialization is supported only for plumbing
smoke tests:

```powershell
python scripts\smoke_test_labram.py --device cpu --random-init
```

## Run the full pipeline

```powershell
cd E:\code\BCI_DayLoop
conda activate bci-dayloop
python scripts\run_pipeline.py --config configs\day1_bnci_s01.yaml
```

The main pipeline assumes the HDF5 file has already been prepared in the
separate data environment. It does not download MOABB data. You can also run
stages separately:

```powershell
python scripts\prepare_bnci2014_001.py --config configs\day1_bnci_s01.yaml
python scripts\inspect_dataset.py data\processed\bnci2014_001_s01.h5
python scripts\smoke_test_labram.py --checkpoint checkpoints\labram-base.pth --device cuda
python scripts\train_linear_probe.py --config configs\day1_bnci_s01.yaml
python scripts\replay_offline.py --config configs\day1_bnci_s01.yaml --max-windows 20
```

`run_pipeline.py` reloads the saved model package in a fresh Python process.
This verifies that inference does not depend on in-memory training objects.

## Streamlit dashboard

```powershell
cd E:\code\BCI_DayLoop
conda activate bci-dayloop
streamlit run web\app.py
```

Or run `run_web.bat`. The page discovers HDF5 files, model packages, model
adapters, and acquirers dynamically. It exposes compute device, confidence,
replay speed, maximum window count, start/stop controls, EEG waveforms,
prediction history, and current/average/P95 latency.

## Tests

```powershell
python -m compileall .
pytest
```

Unit tests use synthetic EEG and a tiny injected encoder; they do not download
BNCI data or require the multi-hundred-megabyte official checkpoint. The main
test/runtime dependency set does not include MOABB.

## Model package contract

`runs/day1_bnci_s01/model_package/` contains:

```text
head.pt
model.yaml
preprocessing.yaml
label_map.json
command_map.json
metrics.json
base_model.json
```

`base_model.json` records the checkpoint path, SHA-256 digest, architecture and
load report. The Base weights are intentionally not duplicated into each run;
keep `checkpoints/labram-base.pth` available when moving or reloading a package.

## Intentionally out of scope

Real EEG devices, Unity/physical vehicle control, LoRA, full fine-tuning,
Kubernetes, MLflow, databases, React, and FastAPI are not implemented.

