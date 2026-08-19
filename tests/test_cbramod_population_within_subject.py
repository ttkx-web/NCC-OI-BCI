from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from bci_dayloop.data.hdf5_dataset import HDF5Metadata


ROOT = Path(__file__).resolve().parents[1]


def load_trainer_module():
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "cbramod_population_head_test",
        ROOT / "scripts" / "train_cbramod_population_head.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cbramod_population_cli_defaults_to_loso_and_resolves_semantics() -> None:
    module = load_trainer_module()
    parser = module.build_argument_parser()
    assert parser.parse_args(["--target-subject", "1"]).split_mode == "loso"

    args = parser.parse_args(
        [
            "--target-subject",
            "1",
            "--split-mode",
            "within-subject",
            "--train-session",
            "source",
            "--test-session",
            "held_out",
            "--validation-ratio",
            "0.2",
            "--class-names",
            "left_hand",
            "right_hand",
            "both_hand",
            "rest",
        ]
    )
    assert args.split_mode == "within-subject"
    assert args.class_names == [
        "left_hand",
        "right_hand",
        "both_hand",
        "rest",
    ]

    metadata = HDF5Metadata(
        sample_rate=250.0,
        channel_names=["C3"],
        class_names=["metadata_0", "metadata_1", "metadata_2", "metadata_3"],
        unit="uV",
        dataset_name="unit-test",
    )
    assert module.resolve_class_names(
        metadata=metadata,
        explicit_class_names=args.class_names,
    ) == ("left_hand", "right_hand", "both_hand", "rest")
