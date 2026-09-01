# Ubuntu server deployment

This document declares the deployment acceptance procedure for the Ubuntu
host. It does not connect to JellyFish or persist EEG.

## Supported target and validation status

- Target operating system: Ubuntu 26.04.
- Target interpreter: Python 3.11.15 in a Miniforge environment.
- Target accelerator: RTX 5080 16 GB with NVIDIA driver 595.71.05.
- Intended GPU runtime: the organization-approved PyTorch 2.12.1 CUDA 13.0
  wheel (`cu130`). The selected wheel index is an infrastructure choice and is
  deliberately not embedded in this repository.

`pyproject.toml` currently declares `>=3.11,<3.12`. The source tree has not
yet been validated in a Python 3.11 environment during this audit, so this is
a deployment target, not a completed Python-3.11 support claim. Do not widen
the range to include Python 3.12 or newer without a separate validation run.

## Create the portable base environment

```bash
git clone <approved-repository-url> NCC-OI-BCI
cd NCC-OI-BCI
conda env create -f environment.yml
conda activate bci-dayloop
python -m pip install --upgrade pip
```

`environment.yml` intentionally contains no PyTorch CUDA toolkit, CUDA wheel
URL, TorchVision, or TorchAudio. First install the organization-approved
server-specific wheel, then install this project without letting pip replace
that wheel:

```bash
python -m pip install --index-url <approved-cu130-wheel-index> \
  'torch==2.12.1+cu130'
python -m pip install -e . --no-deps
```

Optional roles are explicit: `.[data]` adds MOABB data preparation, `.[ui]`
adds pandas and Streamlit, and `.[dev]` adds pytest. They are not required for
the headless realtime probe and benchmark host.

## Hardware acceptance gate

Run the metadata-only probe before any package or device work:

```bash
python scripts/check_runtime_environment.py --require-cuda
python -m compileall src scripts tests
PYTEST_TEMP="$(mktemp -d)"
python -m pytest -q --basetemp="$PYTEST_TEMP"
```

The probe prints Python, platform, torch, CUDA runtime, CUDA availability, GPU
name, compute capability, and NumPy/SciPy/MNE versions. It runs only a tiny
CUDA tensor calculation when CUDA is available; it does not access an EEG
device, load EEG, or emit hardware serial numbers.

The server deployment is accepted only after the Python 3.11 test run passes,
CUDA is available, and the tensor smoke reports `passed`.

## JellyFish configuration boundary

The host is supplied at runtime, never written to a file or deployment
artifact:

```bash
export NEURACLE_JELLYFISH_HOST='<isolated-jellyfish-host>'
```

Do not place an address, password, serial number, or subject identifier in
configuration, summaries, or commits.

## Runtime Package relocation

All twelve formal schema-v2 packages resolve runtime components from relative
paths in `package.yaml`; the loader rejects absolute and escaping component
paths and can verify the declared SHA-256 hashes. Copy each complete package
directory unchanged, then verify it on the server:

```bash
python -c "from bci_dayloop.packages.loader import load_runtime_package; load_runtime_package('<package-dir>', device='cpu', verify_hashes=True); print('verified')"
```

The 1/2/3/4 s LaBraM Live19 and CBRaMod Live19-to-Spline22 packages contain no
Windows absolute paths in their YAML/JSON metadata. The four 50M packages do
contain Windows absolute source paths in `export_manifest.json`; these are
provenance-only export records and are not consumed by the schema-v2 runtime
loader. Do not rewrite frozen package files to remove them, because that would
invalidate the preserved artifact record. The relocatable procedure is to copy
the package directory intact and verify hashes after transfer.

Package transfer and `verify_hashes=True` are required before a server-side
realtime benchmark. No model, preprocessing, package schema, realtime policy,
or benchmark metric changes are part of this deployment procedure.
