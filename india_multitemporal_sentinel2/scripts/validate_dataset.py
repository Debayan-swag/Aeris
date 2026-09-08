"""
Validate the India multi-temporal Sentinel-2 dataset.

Checks file integrity, metadata consistency, temporal ordering, and spatial footprint.
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

from india_multitemporal_sentinel2 import config  # noqa: E402
from india_multitemporal_sentinel2.pipeline_utils import (  # noqa: E402
    load_csv,
    logger,
    read_geotiff_meta,
    setup_logging,
)


def _parse_list(s: str):
    try:
        return ast.literal_eval(s)
    except Exception:  # noqa: BLE001
        return None


def validate() -> tuple[bool, list[str]]:
    setup_logging()
    issues: list[str] = []
    warnings: list[str] = []

    rows = load_csv(config.TEMPORAL_METADATA_CSV)
    if not rows:
        issues.append("temporal_metadata.csv is empty or missing")
        return False, issues

    image_ids = [r["image_id"] for r in rows]
    if len(image_ids) != len(set(image_ids)):
        issues.append("Duplicate image_id values found")

    loc_date = [(r["location_id"], r["acquisition_date"]) for r in rows]
    if len(loc_date) != len(set(loc_date)):
        issues.append("Duplicate location_id + acquisition_date combinations found")

    meta_paths = set()
    for r in rows:
        path = config.PACKAGE_DIR / r["image_path"]
        meta_paths.add(path.resolve())
        if not path.exists():
            issues.append(f"Missing image for metadata row: {r['image_id']} -> {r['image_path']}")
            continue
        try:
            meta = read_geotiff_meta(path)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Unreadable GeoTIFF {path}: {exc}")
            continue
        if meta["count"] != 6:
            issues.append(f"{path.name}: expected 6 bands, got {meta['count']}")
        if not meta["crs"]:
            issues.append(f"{path.name}: missing CRS")
        if meta["transform"] is None:
            issues.append(f"{path.name}: missing transform")
        # Resolution approximately 10m (EPSG:4326 degree scale differs — check if projected)
        res = meta["res"]
        if res and abs(res[0]) > 0:
            # If CRS is geographic, degree resolution ≈ 10m / 111320
            if "4326" in str(meta["crs"]):
                approx_m = abs(res[0]) * 111320
                if not (5 <= approx_m <= 20):
                    warnings.append(
                        f"{path.name}: geographic res ~{approx_m:.1f}m (expected ~10m)"
                    )
            else:
                if not (8 <= abs(res[0]) <= 12):
                    warnings.append(f"{path.name}: resolution {res} (expected ~10m)")

        try:
            cloud = float(r["cloud_percentage"])
        except (TypeError, ValueError):
            issues.append(f"{r['image_id']}: non-numeric cloud_percentage")
            cloud = None
        if cloud is not None and not (0 <= cloud <= config.FALLBACK_CLOUD_LIMIT + 1e-6):
            issues.append(f"{r['image_id']}: cloud_percentage {cloud} outside expected range")

        try:
            datetime.strptime(r["acquisition_date"], "%Y-%m-%d")
        except Exception:  # noqa: BLE001
            issues.append(f"{r['image_id']}: invalid acquisition_date {r['acquisition_date']}")

        if not r.get("product_id") and not r.get("granule_id"):
            warnings.append(f"{r['image_id']}: missing product_id and granule_id")

    # Orphan images
    if config.IMAGES_DIR.exists():
        for path in config.IMAGES_DIR.rglob("*.tif"):
            if path.resolve() not in meta_paths:
                issues.append(f"Image without metadata: {path}")

    # Temporal ordering + footprint consistency per location
    by_loc: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_loc[r["location_id"]].append(r)

    for lid, items in by_loc.items():
        items = sorted(items, key=lambda x: x["acquisition_date"])
        dates = [x["acquisition_date"] for x in items]
        if dates != sorted(dates):
            issues.append(f"{lid}: dates not chronologically sortable")
        for a, b in zip(dates, dates[1:]):
            if not (a < b):
                issues.append(f"{lid}: date order violation {a} !< {b}")

        footprints = []
        for x in items:
            key = (x.get("crs"), x.get("width"), x.get("height"), x.get("transform"), x.get("bbox"))
            footprints.append(key)
        if len(set(footprints)) > 1:
            issues.append(f"{lid}: inconsistent spatial footprint across dates")

    # Write report
    lines = [
        "Validation Report — India Multi-Temporal Sentinel-2",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        f"Rows: {len(rows)}",
        f"Locations with images: {len(by_loc)}",
        "",
        f"ISSUES ({len(issues)}):",
    ]
    lines.extend(f"  - {i}" for i in issues) if issues else lines.append("  (none)")
    lines.append("")
    lines.append(f"WARNINGS ({len(warnings)}):")
    lines.extend(f"  - {w}" for w in warnings) if warnings else lines.append("  (none)")

    # Temporal pairs summary
    lines.append("")
    lines.append("Temporal pairs / days_between_acquisitions:")
    for lid, items in sorted(by_loc.items()):
        items = sorted(items, key=lambda x: x["acquisition_date"])
        for i in range(len(items) - 1):
            d0 = datetime.strptime(items[i]["acquisition_date"], "%Y-%m-%d")
            d1 = datetime.strptime(items[i + 1]["acquisition_date"], "%Y-%m-%d")
            days = (d1 - d0).days
            lines.append(
                f"  {lid}: {items[i]['acquisition_date']} → {items[i+1]['acquisition_date']} ({days} days)"
            )

    config.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    config.VALIDATION_REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Validation %s — %s issues, %s warnings", "PASSED" if not issues else "FAILED", len(issues), len(warnings))
    print("\n".join(lines))
    return len(issues) == 0, issues


def main() -> int:
    ok, _ = validate()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
