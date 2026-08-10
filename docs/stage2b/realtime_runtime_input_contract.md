# Stage 2B Realtime Runtime Input Contract

This document closes the approved realtime mapping and preparation gate. It is
an interface contract only: this gate prepares input but does not invoke model
inference, a classifier head, or a checkpoint.

## CONFIRMED: authoritative target and approved source mapping

The sole authoritative target-channel definition is the unified Runtime
`STANDARD_64_CHANNELS` definition in
`bci_dayloop.models.model_50m.config`, including `F9` and `F10`. The immutable
machine-readable policy is
`ApprovedRealtimeMappingPolicy(policy_id="neuracle_59_to_standard64_v1")`.
It directly references that Runtime definition and fingerprints it so that a
target-channel change fails closed pending policy review.

The approved ordered realtime EEG source is:

```text
Fpz,Fp1,Fp2,AF3,AF4,AF7,AF8,Fz,F1,F2,F3,F4,F5,F6,F7,F8,
FCz,FC1,FC2,FC3,FC4,FC5,FC6,FT7,FT8,Cz,C1,C2,C3,C4,C5,C6,
T7,T8,CP1,CP2,CP3,CP4,CP5,CP6,TP7,TP8,Pz,P3,P4,P5,P6,P7,
P8,POz,PO3,PO4,PO5,PO6,PO7,PO8,Oz,O1,O2
```

It contains 59 channels. The approved policy has exactly 57 same-name target
matches. `PO5` and `PO6` are the only approved ignored source channels. The
only approved missing target channels are `AFz`, `CPz`, `P1`, `P2`, `Iz`, `F9`,
and `F10`; unified Runtime preprocessing may explicitly zero-fill precisely
those positions and make their `channel_valid_mask` values false. Aliases,
duplicates, additional missing targets, and additional unknown sources are not
approved.

## CONFIRMED: source-window gate

Before preparation, a `RealtimeWindow` must contain exactly `[59, 4000]`
samples at 1000 Hz for 4.0 seconds, in the source order above. It must have
4000 strictly increasing timestamps, unit `uV`, `source_unit="uV"`, unit
evidence `vendor_confirmed`, and `model_safe=true`. Its channel types must all
be EEG and its per-channel units must all be `uV`; all provenance lengths must
match the channel count. A continuous-segment identifier is mandatory and
samples must be finite.

Source `model_safe` means only that the realtime source unit/channel contract
passed. It is not the same as final model-input safety.

## CONFIRMED: one formal preparation entry

`RealtimeRuntimeBridge` is the only Stage 2B bridge in this phase:

```text
RealtimeWindow → source gate → RawEEGWindow → RuntimeModel.prepare()
               → prepared-input gate → RealtimePreparedWindow
```

The adapter retains the exact source samples, source channel order, sampling
rate, unit, window ID, continuous-segment provenance, and marker summaries. It
does not reorder, remove `PO5`/`PO6`, zero-fill, resample, filter, reference,
normalize, crop, pad, call `predict`, call `predict_prepared`, or call a
classifier head. `RuntimeModel.prepare()` remains the single owner of
`SignalCanonicalizer`, `Model50MInputTransform`, and Runtime Package input
contract processing. Runtime Package / unified Runtime preprocessing
configuration is the sole authority for preprocessing parameters.

## CONFIRMED: prepared 50M contract

The approved Runtime Package output is finite `torch.float32` `signal` with
shape `[1, 64, 400]`, and `channel_valid_mask` with shape `[1, 64]` containing
only boolean or 0/1 values. It must contain exactly 57 valid positions. The
false positions must be exactly the seven approved missing target-channel
positions above. Runtime diagnostics must report only `PO5`/`PO6` as unknown
source channels, 57 mapped channels, seven missing target channels, zero
duplicates, and zero padded or cropped points.

The Runtime Package contract must declare the authoritative 64-channel order,
100 Hz, 4.0 seconds, 400 samples, `uV`, `("signal", "channel_valid_mask")`,
`output_layer_idx=8`, and `aggregation="flatten"`. The approved 4-second real
checkpoint/head contract is `[1,64,400]` with this mask and aggregation.

## BLOCKED / fail-closed conditions

The bridge returns `model_input_safe=false` and does not expose a prepared
input if any source, Runtime Package, signal, mask, or diagnostics condition
above differs. It performs no remediation: it cannot silently reorder, drop,
pad, interpolate, synthesize, resize, or average channels.

Inference remains intentionally blocked in this phase. A separate explicitly
authorized step is required before any call to Runtime prediction or a model
head.
