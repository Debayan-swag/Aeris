"""Shared helpers for the India multi-temporal Sentinel-2 pipeline."""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger("india_s2")


def setup_logging(level: int = logging.INFO) -> None:
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(level)


def utm_epsg_from_lon_lat(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) / 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def fixed_aoi_geojson(lon: float, lat: float, width_m: float, height_m: float) -> dict:
    """
    Build a fixed square AOI centered on (lon, lat) in a local UTM CRS,
    then return a WGS84 GeoJSON Polygon for Earth Engine.
    """
    from pyproj import Transformer

    epsg = utm_epsg_from_lon_lat(lon, lat)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_wgs = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    x, y = to_utm.transform(lon, lat)
    half_w = width_m / 2.0
    half_h = height_m / 2.0
    corners_utm = [
        (x - half_w, y - half_h),
        (x + half_w, y - half_h),
        (x + half_w, y + half_h),
        (x - half_w, y + half_h),
        (x - half_w, y - half_h),
    ]
    coords = [list(to_wgs.transform(cx, cy)) for cx, cy in corners_utm]
    return {"type": "Polygon", "coordinates": [coords], "crs_epsg": epsg}


def aoi_bounds_wgs84(aoi: dict) -> tuple[float, float, float, float]:
    coords = aoi["coordinates"][0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return min(xs), min(ys), max(xs), max(ys)


def retry_call(fn, *, retries: int, base_delay: float, what: str):
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - continue other locations
            last_exc = exc
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning("%s failed (attempt %s/%s): %s", what, attempt, retries, exc)
            if attempt < retries:
                time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def ms_to_iso(ms: int | float) -> tuple[str, str, int, int]:
    """Return (date, datetime_iso, year, month) from EE system:time_start ms."""
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m-%dT%H:%M:%SZ"), dt.year, dt.month


def quality_flag(cloud_pct: float, primary: float, fallback: float) -> str:
    if cloud_pct <= primary:
        return "good"
    if cloud_pct <= fallback:
        return "acceptable_fallback"
    return "too_cloudy"


def select_temporal_scenes(
    scenes: list[dict],
    target_n: int,
) -> list[dict]:
    """
    Distribute real scenes across the available temporal range.

    scenes: list of dicts with keys acquisition_date (YYYY-MM-DD), cloud_percentage,
            and any EE metadata already attached. Must already be unique by date.
    """
    if not scenes:
        return []
    scenes = sorted(scenes, key=lambda s: s["acquisition_date"])
    if len(scenes) <= target_n:
        return scenes

    # Deduplicate by date keeping lowest cloud
    by_date: dict[str, dict] = {}
    for s in scenes:
        d = s["acquisition_date"]
        if d not in by_date or s["cloud_percentage"] < by_date[d]["cloud_percentage"]:
            by_date[d] = s
    scenes = sorted(by_date.values(), key=lambda s: s["acquisition_date"])
    if len(scenes) <= target_n:
        return scenes

    n = len(scenes)
    # Divide chronological index space into target_n intervals
    selected: list[dict] = []
    used_idx: set[int] = set()
    for i in range(target_n):
        start = int(math.floor(i * n / target_n))
        end = int(math.floor((i + 1) * n / target_n))
        end = max(end, start + 1)
        window = list(range(start, min(end, n)))
        # Prefer lowest cloud in window
        best = min(window, key=lambda idx: (scenes[idx]["cloud_percentage"], scenes[idx]["acquisition_date"]))
        if best in used_idx:
            # Search neighbors
            for offset in range(1, n):
                for cand in (best - offset, best + offset):
                    if 0 <= cand < n and cand not in used_idx:
                        best = cand
                        break
                else:
                    continue
                break
        used_idx.add(best)
        selected.append(scenes[best])

    return sorted(selected, key=lambda s: s["acquisition_date"])


def read_geotiff_meta(path: Path) -> dict[str, Any]:
    import rasterio

    with rasterio.open(path) as ds:
        transform = ds.transform
        bounds = ds.bounds
        return {
            "width": ds.width,
            "height": ds.height,
            "crs": str(ds.crs) if ds.crs else "",
            "transform": list(transform)[:6],
            "bbox": [bounds.left, bounds.bottom, bounds.right, bounds.top],
            "count": ds.count,
            "res": ds.res,
        }


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_csv_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b) with divide-by-zero → nan."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = a + b
    out = np.full(a.shape, np.nan, dtype=np.float32)
    mask = np.abs(denom) > 1e-6
    out[mask] = (a[mask] - b[mask]) / denom[mask]
    return out


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
