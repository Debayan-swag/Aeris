"""
Compute NDVI / NDWI / NDBI from multispectral GeoTIFFs.

Does not modify source imagery.
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
from india_multitemporal_sentinel2.spectral import (  # noqa: E402
    compute_indices,
    mean_valid,
    write_index_raster,
)


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Compute NDVI/NDWI/NDBI")
    parser.add_argument("--image-id", default="", help="image_id from temporal_metadata.csv")
    parser.add_argument("--path", default="", help="Direct path to a GeoTIFF")
    parser.add_argument("--write", action="store_true", help="Write index GeoTIFFs")
    parser.add_argument("--out-dir", default="", help="Output directory for index rasters")
    args = parser.parse_args()

    if args.path:
        path = Path(args.path)
        image_id = path.stem
    elif args.image_id:
        rows = load_csv(config.TEMPORAL_METADATA_CSV)
        match = [r for r in rows if r["image_id"] == args.image_id]
        if not match:
            logger.error("image_id not found: %s", args.image_id)
            return 1
        path = config.PACKAGE_DIR / match[0]["image_path"]
        image_id = args.image_id
    else:
        logger.error("Provide --image-id or --path")
        return 1

    if not path.exists():
        logger.error("File not found: %s", path)
        return 1

    indices = compute_indices(path)
    for name, arr in indices.items():
        logger.info(
            "%s %s mean=%.4f min=%.4f max=%.4f",
            image_id,
            name,
            mean_valid(arr),
            float(np.nanmin(arr)),
            float(np.nanmax(arr)),
        )

    if args.write:
        out_dir = Path(args.out_dir) if args.out_dir else path.parent / "indices"
        for name, arr in indices.items():
            out = out_dir / f"{path.stem}_{name}.tif"
            write_index_raster(path, arr, out)
            logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
