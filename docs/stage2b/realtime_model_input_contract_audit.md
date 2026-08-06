# Stage 2B Realtime Model Input Contract Audit

## Scope and safety decision

This is an audit and fail-closed boundary only. It does not load a checkpoint, resample, filter, re-reference, normalize, map channels, pad, interpolate, synthesize channels, or invoke inference.

The verified realtime Source window is currently 59 EEG channels at 1000 Hz for 4 seconds (`[59, 4000]`), in `uV`, with `unit_evidence_level=vendor_confirmed`. Its existing `model_safe=true` means only that the channel-type and unit contract passed; it is not model-input approval.

`audit_realtime_window_model_input()` is the new non-mutating gate. It accepts a `RealtimeWindow` and requires an exact model-shaped window. It only reports differences; it never silently reorders, drops, fills, resizes, or transforms samples.

## Confirmed from repository code and tests

### Stage 2A shared preprocessor

- `src/bci_dayloop/data/preprocessing.py` defines the shared training/replay preprocessor, not a Stage 2B runtime bridge.
- Its configured defaults are 0.1–75 Hz fourth-order band-pass, 50 Hz notch, polyphase resampling to 200 Hz, conversion to `uV`, and per-window/per-channel Z-score with epsilon `1e-6`.
- It selects EEG channels but does not define a fixed 64-channel montage or `channel_valid_mask`; it also truncates a remainder to the configured patch size when reshaping.
- Therefore this preprocessor is **not** evidence that a 59-channel, 1000 Hz realtime window can enter the 50M model unchanged.

### 50M model-side configuration

- `src/bci_dayloop/models/model_50m/config.py` confirms the code-defined target montage has 64 names in a fixed order, default target rate 100 Hz, patch length/stride 1 s, `output_layer_idx=8`, and `aggregation=flatten`.
- For an explicitly configured 4.0 s window, those formulas yield signal `[64, 400]`, 4 time patches per channel, tokens `[256, 100]`, and flatten feature size `[1, 131072]`. This matches `docs/contracts/stage1_stage2a_4s_handoff.md`.
- The model-side preprocessor accepts finite numeric `[C,T]` input and explicit `V`/`mV`/`uV`; it converts to `uV`, maps to its fixed 64 channels, optionally applies average reference (default `none`), applies a fourth-order 0.1–75 Hz zero-phase band-pass by default, uses `scipy.signal.resample_poly` to 100 Hz, and applies per-valid-channel time-axis Z-score with epsilon `1e-8`.
- The current 50M code has no notch setting. NaN or Inf input is rejected before conversion.
- Its implementation can ignore unknown channels, average duplicate mapped channels, fill missing channels with 0, and crop or pad output time points. Those are current model-preprocessor behaviors, **not permitted behaviors at the Stage 2B realtime gate**.

## Inferred from the code-defined four-second configuration

The `[64,400]`, `[256,100]`, and `[1,131072]` values follow directly from setting `window_seconds=4.0` and `target_sample_rate=100.0` in the code-defined configuration. They are deterministic configuration results, not evidence that a real deployed checkpoint or classifier head has been loaded and validated with that four-second setting.

## Confirmed strict Stage 2B gate

The new gate uses the existing code-defined `STANDARD_64_CHANNELS` and `CHANNEL_ALIASES` from `src/bci_dayloop/models/model_50m/config.py`; it does not introduce a second channel mapping configuration.

Before a realtime window could be considered model-input safe, it must already have all of the following:

- exact shape `[64, 400]` and 4.0 seconds at 100 Hz;
- exact `uV` unit;
- exact code-defined 64 channel names in code-defined order;
- no duplicate, alias-only, case-only, missing, or unexpected names;
- strictly increasing timestamps;
- `metadata.model_safe=true`, `unit_evidence_level=vendor_confirmed`, and per-channel `channel_types=[eeg,…]`, `channel_units=[uV,…]` with lengths matching the samples.

Comparison-only alias and case reporting uses the existing `CHANNEL_ALIASES`; it never changes an input name or its position. Any difference keeps `model_input_safe=false`.

## Current 59-channel comparison

### Confirmed

- The realtime window has 59 EEG channels, 1000 Hz, 4000 samples, and 4 seconds.
- The strict 4-second model boundary is 64 channels, 100 Hz, and 400 samples.
- Thus the direct realtime window fails the exact channel-count, sampling-rate, and sample-count checks. It cannot be passed to a model.

### Unresolved

- The anonymized repository contains no persisted, authoritative ordered list of the 59 live EEG channel names. A true per-name report of matched channels, missing model channels, unexpected channels, aliases, case differences, order differences, and duplicate names cannot be produced without recording or supplying that approved metadata.
- No existing verified 59-to-64 realtime mapping configuration was found. This audit therefore does not design one.
- `docs/standard_64_channels.json` ends with `A1`, `A2`, while the code-defined 50M `STANDARD_64_CHANNELS` ends with `F9`, `F10`. The baseline must be resolved against approved checkpoint/package metadata before a mapping can be verified.
- No real checkpoint/package metadata was loaded in this audit. The repository's default 50M Runtime remains a 10-second configuration; code can express 4 seconds, but this audit found no repository test proving a real checkpoint was validated under the four-second configuration.
- The 50M reference requirement is not checkpoint-confirmed: code default is `none`, while average reference is an optional alternative.

### Blocked

- The current 59-channel/1000 Hz/[59,4000] `RealtimeWindow` is **not model-input safe**.
- The current window assembler does not persist `channel_types`, `channel_units`, `unit_evidence_level`, or `model_safe` into generated `RealtimeWindow.metadata`; the gate therefore also rejects an assembled live window until provenance propagation is explicitly designed and tested.
- No preprocessing or model call is authorized until the 59-to-target mapping, channel baseline, 1000-to-100 Hz resampling contract, reference, filters, normalization, and checkpoint-specific four-second compatibility are resolved and independently validated.

## Evidence classification

- **Confirmed:** facts directly implemented in the cited repository code/tests and the documented realtime source/window validation.
- **Inferred:** the `[64,400]`, token, and feature sizes are deterministic formulas for an explicitly selected 4-second `Model50MConfig`; they are not proof of a real checkpoint deployment.
- **Unresolved:** fields requiring approved live channel metadata, checkpoint/package metadata, or a validation artifact not present in the repository.
- **Blocked:** actions that would transform the live 59-channel window or invoke a model before those unresolved items are closed.

This document intentionally contains no device identifier, network address, participant data, raw EEG, or absolute device timestamp.
