from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import pytest

from bci_dayloop.data import bnci

MOABB_AVAILABLE = find_spec("moabb") is not None


@pytest.mark.skipif(
    bnci.os.name != "nt" or not MOABB_AVAILABLE,
    reason="Windows drive-letter workaround requires optional MOABB data dependencies",
)
def test_moabb_sanitizer_preserves_windows_drive():
    from moabb.datasets import download as moabb_download

    bnci._apply_moabb_windows_path_fix()
    result = moabb_download._sanitize_path(Path(r"E:\cache\bad:name.mat"))
    assert result.drive.upper() == "E:"
    assert result.name == "bad-name.mat"
