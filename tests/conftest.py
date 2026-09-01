"""Test-only environment defaults for optional desktop-science libraries."""

from __future__ import annotations

import os


# MNE otherwise creates a lock below the invoking user's home directory during
# import.  Collection must not require write access outside the test sandbox.
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
