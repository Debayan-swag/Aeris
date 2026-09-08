"""
Rank locations by observed multi-temporal change (real raster differences).

location_type is NOT used as evidence of change — only measured indices / spectral diffs.
Writes metadata/change_candidates.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

from india_multitemporal_sentinel2 import config  # noqa: E402
from india_multitemporal_sentinel2.pipeline_utils import (  # noqa: E402
    load_csv,
    logger,
    setup_logging,
    write_csv,
)
from india_multitemporal_sentinel2.spectral import (  # noqa: E402
    compute_indices,
    mean_valid,
    read_bands,
)


def spectral_mean_abs_diff(path_a: Path, path_b: Path) -> float:
    a = read_bands(path_a, config.BANDS)
    b = read_bands(path_b, config.BANDS)
    diffs = []
    for name in config.BANDS:
        # Reflectance scale often 0-10000; normalize lightly
        da = a[name]
        db = b[name]
        mask = np.isfinite(da) & np.isfinite(db) & (da > 0) & (db > 0)
        if not mask.any():
            continue
        diffs.append(float(np.mean(np.abs(da[mask] - db[mask]) / 10000.0)))
    if not diffs:
        return float("nan")
    return float(np.mean(diffs))


def pair_change(path_a: Path, path_b: Path) -> dict:
    ia = compute_indices(path_a)
    ib = compute_indices(path_b)
    ndvi_c = mean_valid(ib["NDVI"]) - mean_valid(ia["NDVI"])
    ndwi_c = mean_valid(ib["NDWI"]) - mean_valid(ia["NDWI"])
    ndbi_c = mean_valid(ib["NDBI"]) - mean_valid(ia["NDBI"])
    spec = spectral_mean_abs_diff(path_a, path_b)
    overall = float(
        np.nanmean(
            [
                abs(ndvi_c),
                abs(ndwi_c),
                abs(ndbi_c),
                spec if np.isfinite(spec) else np.nan,
            ]
        )
    )
    return {
        "ndvi_change": round(ndvi_c, 6),
        "ndwi_change": round(ndwi_c, 6),
        "ndbi_change": round(ndbi_c, 6),
        "spectral_change_score": round(spec, 6) if np.isfinite(spec) else "",
        "overall_change_score": round(overall, 6) if np.isfinite(overall) else "",
    }


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Generate change_candidates.csv")
    parser.add_argument(
        "--pair-mode",
        default="endpoints",
        choices=["endpoints", "max_adjacent"],
        help="endpoints=first vs last date; max_adjacent=best consecutive pair",
    )
    args = parser.parse_args()

    rows = load_csv(config.TEMPORAL_METADATA_CSV)
    by_loc: dict[str, list[dict]] = {}
    for r in rows:
        by_loc.setdefault(r["location_id"], []).append(r)

    candidates = []
    for lid, items in sorted(by_loc.items()):
        items = sorted(items, key=lambda x: x["acquisition_date"])
        if len(items) < 2:
            continue
        city = items[0]["city"]
        if args.pair_mode == "endpoints":
            pairs = [(items[0], items[-1])]
        else:
            pairs = list(zip(items, items[1:]))

        best = None
        for a, b in pairs:
            path_a = config.PACKAGE_DIR / a["image_path"]
            path_b = config.PACKAGE_DIR / b["image_path"]
            if not path_a.exists() or not path_b.exists():
                continue
            try:
                metrics = pair_change(path_a, path_b)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s pair failed: %s", lid, exc)
                continue
            cand = {
                "location_id": lid,
                "city": city,
                "date_before": a["acquisition_date"],
                "date_after": b["acquisition_date"],
                **metrics,
            }
            if best is None or float(cand["overall_change_score"] or 0) > float(
                best["overall_change_score"] or 0
            ):
                best = cand
        if best:
            candidates.append(best)

    candidates.sort(key=lambda c: float(c["overall_change_score"] or 0), reverse=True)
    write_csv(
        config.CHANGE_CANDIDATES_CSV,
        candidates,
        [
            "location_id",
            "city",
            "date_before",
            "date_after",
            "ndvi_change",
            "ndwi_change",
            "ndbi_change",
            "spectral_change_score",
            "overall_change_score",
        ],
    )
    logger.info("Wrote %s change candidates → %s", len(candidates), config.CHANGE_CANDIDATES_CSV)
    for c in candidates[:10]:
        logger.info(
            "  %s %s→%s score=%s",
            c["location_id"],
            c["date_before"],
            c["date_after"],
            c["overall_change_score"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
