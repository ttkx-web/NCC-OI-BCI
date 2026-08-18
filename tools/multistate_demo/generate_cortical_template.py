#!/usr/bin/env python3
"""Generate static fsaverage5 cortical PNG assets for the multi-state demo.

This is a development-only asset utility.  It intentionally lives outside the
project's production scripts because Nilearn is not a runtime dependency of the
Streamlit demo.  The generated images support a sensor-derived cortical
activity visualization; they are not EEG source-localization outputs.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "src" / "bci_dayloop" / "demo" / "assets" / "cortical"
TEMPLATE_WIDTH = 512
TEMPLATE_HEIGHT = 384


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate static cortical templates for the multi-state demo.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=TEMPLATE_WIDTH)
    parser.add_argument("--height", type=int, default=TEMPLATE_HEIGHT)
    parser.add_argument("--dpi", type=int, default=144)
    return parser.parse_args()


def _crop_and_center(source: Path, destination: Path, *, width: int, height: int) -> None:
    with Image.open(source) as rendered:
        image = rendered.convert("RGBA")
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        raise RuntimeError(f"Cortical rendering {source} contains no visible pixels")
    cropped = image.crop(bbox)
    padding = max(4, min(width, height) // 45)
    scale = min((width - 2 * padding) / cropped.width, (height - 2 * padding) / cropped.height)
    target_size = (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale)))
    cropped = cropped.resize(target_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    offset = ((width - cropped.width) // 2, (height - cropped.height) // 2)
    canvas.alpha_composite(cropped, offset)
    if canvas.mode != "RGBA" or canvas.getchannel("A").getextrema()[0] != 0:
        raise RuntimeError("Expected an RGBA cortical template with a transparent background")
    canvas.save(destination)


def render_template(*, mesh: str, sulc: str, hemi: str, destination: Path, width: int, height: int, dpi: int) -> None:
    # Nilearn is deliberately imported only by this one-off generator.
    import matplotlib.pyplot as plt
    from nilearn import plotting

    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=(0, 0, 0, 0))
    axes = figure.add_subplot(111, projection="3d")
    axes.set_facecolor((0, 0, 0, 0))
    plotting.plot_surf(
        surf_mesh=mesh,
        bg_map=sulc,
        hemi=hemi,
        view="lateral",
        engine="matplotlib",
        colorbar=False,
        cmap="gray",
        axes=axes,
        figure=figure,
    )
    axes.set_axis_off()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        figure.savefig(temporary_path, dpi=dpi, transparent=True, facecolor=(0, 0, 0, 0), pad_inches=0)
        _crop_and_center(temporary_path, destination, width=width, height=height)
    finally:
        temporary_path.unlink(missing_ok=True)
        plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.dpi <= 0:
        raise SystemExit("--width, --height and --dpi must be positive")
    from nilearn import datasets

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    outputs = {
        "left": args.output_dir / "cortical_left_lateral.png",
        "right": args.output_dir / "cortical_right_lateral.png",
    }
    render_template(
        mesh=str(fsaverage.pial_left), sulc=str(fsaverage.sulc_left), hemi="left",
        destination=outputs["left"], width=args.width, height=args.height, dpi=args.dpi,
    )
    render_template(
        mesh=str(fsaverage.pial_right), sulc=str(fsaverage.sulc_right), hemi="right",
        destination=outputs["right"], width=args.width, height=args.height, dpi=args.dpi,
    )
    for side, path in outputs.items():
        with Image.open(path) as image:
            if image.mode != "RGBA" or image.size != (args.width, args.height) or image.getchannel("A").getextrema()[0] != 0:
                raise RuntimeError(f"Invalid {side} cortical template: {path}")
        print(f"Generated {path.relative_to(ROOT)} ({args.width}x{args.height}, RGBA transparent background)")


if __name__ == "__main__":
    main()
