"""NDVI / NDWI / NDBI helpers (source GeoTIFFs are never modified)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from india_multitemporal_sentinel2.pipeline_utils import safe_index

BAND_INDEX = {"B2": 1, "B3": 2, "B4": 3, "B8": 4, "B11": 5, "B12": 6}


def read_bands(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    import rasterio

    out = {}
    with rasterio.open(path) as ds:
        for name in names:
            out[name] = ds.read(BAND_INDEX[name]).astype(np.float32)
    return out


def compute_indices(path: Path) -> dict[str, np.ndarray]:
    bands = read_bands(path, ["B3", "B4", "B8", "B11"])
    return {
        "NDVI": safe_index(bands["B8"], bands["B4"]),
        "NDWI": safe_index(bands["B3"], bands["B8"]),
        "NDBI": safe_index(bands["B11"], bands["B8"]),
    }


def mean_valid(arr: np.ndarray) -> float:
    m = np.isfinite(arr)
    if not m.any():
        return float("nan")
    return float(np.nanmean(arr))


def write_index_raster(source: Path, array: np.ndarray, out_path: Path) -> None:
    import rasterio

    with rasterio.open(source) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype="float32", nodata=np.nan)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(array.astype(np.float32), 1)
