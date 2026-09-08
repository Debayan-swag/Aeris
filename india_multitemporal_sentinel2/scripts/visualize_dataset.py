"""
Visualize temporal sequences for a city / location_id.

Supports true-color, false-color, NDVI/NDWI/NDBI, and before/after panels.
Does not modify source GeoTIFFs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

from india_multitemporal_sentinel2 import config  # noqa: E402
from india_multitemporal_sentinel2.pipeline_utils import load_csv, logger, setup_logging  # noqa: E402
from india_multitemporal_sentinel2.spectral import compute_indices, read_bands  # noqa: E402


def _percentile_stretch(rgb: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=np.float32)
    for i in range(rgb.shape[2]):
        band = rgb[:, :, i]
        lo, hi = np.nanpercentile(band, [p_low, p_high])
        if hi <= lo:
            hi = lo + 1.0
        out[:, :, i] = np.clip((band - lo) / (hi - lo), 0, 1)
    return out


def _to_display(mode: str, path: Path) -> np.ndarray:
    if mode == "truecolor":
        b = read_bands(path, ["B4", "B3", "B2"])
        rgb = np.dstack([b["B4"], b["B3"], b["B2"]])
        return _percentile_stretch(rgb)
    if mode == "falsecolor":
        b = read_bands(path, ["B8", "B4", "B3"])
        rgb = np.dstack([b["B8"], b["B4"], b["B3"]])
        return _percentile_stretch(rgb)
    indices = compute_indices(path)
    arr = indices[mode.upper()]
    # map [-1,1]-ish to a simple colormap via matplotlib
    return arr


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Visualize temporal Sentinel-2 sequence")
    parser.add_argument("--city", default="")
    parser.add_argument("--location-id", required=True)
    parser.add_argument(
        "--mode",
        default="truecolor",
        choices=["truecolor", "falsecolor", "NDVI", "NDWI", "NDBI", "before_after"],
    )
    parser.add_argument("--out", default="", help="Output PNG path")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    rows = [
        r
        for r in load_csv(config.TEMPORAL_METADATA_CSV)
        if r["location_id"] == args.location_id
        and (not args.city or r["city"].lower() == args.city.lower())
    ]
    rows = sorted(rows, key=lambda r: r["acquisition_date"])
    if not rows:
        logger.error("No images for %s", args.location_id)
        return 1

    import matplotlib.pyplot as plt

    if args.mode == "before_after":
        if len(rows) < 2:
            logger.error("Need >=2 images for before/after")
            return 1
        paths = [config.PACKAGE_DIR / rows[0]["image_path"], config.PACKAGE_DIR / rows[-1]["image_path"]]
        labels = [rows[0]["acquisition_date"], rows[-1]["acquisition_date"]]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, path, label in zip(axes, paths, labels):
            ax.imshow(_to_display("truecolor", path))
            ax.set_title(f"{args.location_id}\n{label}")
            ax.axis("off")
    else:
        n = len(rows)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
        if n == 1:
            axes = [axes]
        for ax, row in zip(axes, rows):
            path = config.PACKAGE_DIR / row["image_path"]
            disp = _to_display(args.mode, path)
            if args.mode in {"NDVI", "NDWI", "NDBI"}:
                im = ax.imshow(disp, cmap="RdYlGn", vmin=-0.5, vmax=0.8)
                fig.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.imshow(disp)
            ax.set_title(row["acquisition_date"], fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{args.location_id} — {args.mode}", fontsize=12)

    fig.tight_layout()
    out = Path(args.out) if args.out else config.METADATA_DIR / "previews" / f"{args.location_id}_{args.mode}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    logger.info("Wrote %s", out)
    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
