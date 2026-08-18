from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "DeviceReader": (".device_reader", "DeviceReader"),
    "DeviceReaderFactory": (".device_reader", "DeviceReaderFactory"),
    "register_builtin_readers": (".device_reader", "register_builtin_readers"),
    "align_events_with_csv": (".event_alignment", "align_events_with_csv"),
    "EEGHDF5": (".hdf5_dataset", "EEGHDF5"),
    "HDF5Metadata": (".hdf5_dataset", "HDF5Metadata"),
    "write_hdf5": (".hdf5_dataset", "write_hdf5"),
    "SequentialDataset": (".sequential_dataset", "SequentialDataset"),
    "SequentialDatasetMetadata": (
        ".sequential_dataset",
        "SequentialDatasetMetadata",
    ),
    "load_sequential_dataset": (
        ".sequential_dataset",
        "load_sequential_dataset",
    ),
    "validate_package_window_contract": (
        ".sequential_dataset",
        "validate_package_window_contract",
    ),
    "WorkloadHDF5": (".workload", "WorkloadHDF5"),
    "prepare_workload_subject": (".workload", "prepare_workload_subject"),
    "NeuracleBDFReader": (".neuracle_bdf", "NeuracleBDFReader"),
    "annotations_to_events": (".neuracle_bdf", "annotations_to_events"),
    "parse_neuracle_marker": (".neuracle_bdf", "parse_neuracle_marker"),
    "EEGPreprocessor": (".preprocessing", "EEGPreprocessor"),
    "PreprocessingConfig": (".preprocessing", "PreprocessingConfig"),
    "EEGEvent": (".records", "EEGEvent"),
    "RawEEGRecord": (".records", "RawEEGRecord"),
    "UnitEvidence": (".records", "UnitEvidence"),
}


__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
