"""NCC BCI Web Console API package."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "src"

# Console workers do not need to persist user-level MNE preferences. Keeping
# MNE configuration ephemeral also makes the API safe in read-only service
# accounts and containers.
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(Path(tempfile.gettempdir()) / "ncc-bci-console-mne"))

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
