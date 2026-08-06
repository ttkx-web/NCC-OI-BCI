"""Protocol-neutral primitives for Stage 2B realtime EEG ingestion."""

from .contracts import EEGChunk, EventMarker, RealtimeWindow, WindowResult

__all__ = ["EEGChunk", "EventMarker", "RealtimeWindow", "WindowResult"]
