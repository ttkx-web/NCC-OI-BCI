from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _exporter_module() -> object:
    sys.path.insert(0, str(SCRIPTS))
    try:
        path = SCRIPTS / "export_labram_model_package.py"
        spec = importlib.util.spec_from_file_location(
            "test_export_labram_model_package_script",
            path,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


def test_workload_package_id_uses_real_metadata() -> None:
    module = _exporter_module()
    package_id = module.build_labram_package_id(
        dataset_name="workload_pbci_hackathon",
        metadata={
            "target_subject": "P01",
            "subject": 99,
            "window_seconds": 2.0,
        },
    )

    assert package_id == (
        "labram_workload_pbci_hackathon_"
        "subject_01_population_2s"
    )
    assert "bnci2014_001" not in package_id
    assert not package_id.endswith("_4s")


def test_bnci_package_id_uses_bnci_metadata_and_subject_fallback() -> None:
    module = _exporter_module()
    package_id = module.build_labram_package_id(
        dataset_name="bnci2014_001",
        metadata={
            "subject": 3,
            "window_seconds": 4.0,
        },
    )

    assert package_id == (
        "labram_bnci2014_001_subject_03_population_4s"
    )
