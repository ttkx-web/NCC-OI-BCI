# Neuracle JellyFish vendor boundary

Source repository: `https://github.com/ttkx-web/oi-armi`
Source commit: `61b7b855376814c1c427871d6d3d623e8e49e9e1`
Source path: `oi-mi/collect/neuracle_api.py`

Internal permission to reuse this source has been confirmed. The copied
`neuracle_api.py` retains the original Neuracle copyright header and author
information. Its protocol constants, binary parsing, and field meanings are
preserved.

## NCC-OI-BCI minimal changes

Only the following wrapper-support changes were made to the vendor file:

1. Added a bounded `updateQueue`, `appendUpdatePacket()`, and
   `getUpdatePacket()`. Each item keeps `samples`, `startTimeStamp`,
   `timeStampLength`, packet range, and `hostReceivedAtMonotonic`. A full queue
   raises an explicit `RuntimeError`; it never silently overwrites an
   unconsumed packet.
2. Enqueued updates after the original bulk and per-module assembly paths have
   emitted data to their existing buffers. The original `RingBuffer`,
   `DoubleBuffer`, and protocol parsing remain in place.
3. Retained read/resolve thread references. `stop()` clears the update queue
   and boundedly joins non-current receiver threads so an upper-layer
   disconnect can release them.

`backend.py` is the only project wrapper that accesses `DataServerThread`. It
exposes anonymized META and timestamp-associated packets to
`realtime/neuracle_jellyfish.py`. No reference GUI, model, logging, or legacy
experiment business logic was imported.
