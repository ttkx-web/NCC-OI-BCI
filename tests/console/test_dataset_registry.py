from __future__ import annotations

import json
from pathlib import Path

from app.services.dataset_service import DatasetRegistry


def test_dataset_discovery_returns_metadata_without_local_path(dataset_file: Path) -> None:
    registry = DatasetRegistry(dataset_file.parents[2])
    items = registry.list()
    assert len(items) == 1
    assert items[0].id == "bnci2014_001"
    assert items[0].subject_id == "S01"
    assert items[0].sessions == ["1test"]
    assert items[0].trial_count == 2
    payload = json.dumps(items[0].model_dump())
    assert str(dataset_file) not in payload
    assert ":\\" not in payload

