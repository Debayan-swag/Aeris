"""Build / regenerate CSV metadata and statistics from downloaded GeoTIFFs."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from india_multitemporal_sentinel2 import config
from india_multitemporal_sentinel2.pipeline_utils import (
    aoi_bounds_wgs84,
    fixed_aoi_geojson,
    load_csv,
    logger,
    read_geotiff_meta,
    setup_logging,
    write_csv,
)

TEMPORAL_FIELDS = [
    "image_id",
    "location_id",
    "city",
    "latitude",
    "longitude",
    "acquisition_date",
    "acquisition_datetime",
    "year",
    "month",
    "sensor",
    "collection",
    "product_id",
    "granule_id",
    "mgrs_tile",
    "cloud_percentage",
    "image_path",
    "bands",
    "resolution",
    "width",
    "height",
    "crs",
    "transform",
    "bbox",
    "quality_flag",
]


def write_locations_csv() -> None:
    rows = []
    for loc in config.LOCATIONS:
        aoi = fixed_aoi_geojson(
            loc["longitude"], loc["latitude"], config.AOI_WIDTH_M, config.AOI_HEIGHT_M
        )
        minx, miny, maxx, maxy = aoi_bounds_wgs84(aoi)
        rows.append(
            {
                "location_id": loc["location_id"],
                "city": loc["city"],
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "location_type": loc["location_type"],
                "aoi_width_m": config.AOI_WIDTH_M,
                "aoi_height_m": config.AOI_HEIGHT_M,
                "target_resolution_m": config.RESOLUTION,
                "aoi_min_lon": minx,
                "aoi_min_lat": miny,
                "aoi_max_lon": maxx,
                "aoi_max_lat": maxy,
                "utm_epsg": aoi["crs_epsg"],
            }
        )
    write_csv(
        config.LOCATIONS_CSV,
        rows,
        [
            "location_id",
            "city",
            "latitude",
            "longitude",
            "location_type",
            "aoi_width_m",
            "aoi_height_m",
            "target_resolution_m",
            "aoi_min_lon",
            "aoi_min_lat",
            "aoi_max_lon",
            "aoi_max_lat",
            "utm_epsg",
        ],
    )


def _parse_date(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return None


def build_statistics(rows: list[dict]) -> tuple[list[dict], dict]:
    by_city: dict[str, list[dict]] = defaultdict(list)
    by_loc: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_city[r["city"]].append(r)
        by_loc[r["location_id"]].append(r)

    city_rows = []
    for city, items in sorted(by_city.items()):
        locs = {i["location_id"] for i in items}
        dates = sorted(i["acquisition_date"] for i in items if i.get("acquisition_date"))
        clouds = []
        for i in items:
            try:
                clouds.append(float(i["cloud_percentage"]))
            except (TypeError, ValueError):
                pass
        pairs = 0
        insuf = 0
        for lid in locs:
            n = len(by_loc[lid])
            pairs += max(0, n - 1)
            if n < config.TARGET_IMAGES_PER_LOCATION:
                insuf += 1
        city_rows.append(
            {
                "city": city,
                "number_of_locations": len(locs),
                "number_of_images": len(items),
                "earliest_date": dates[0] if dates else "",
                "latest_date": dates[-1] if dates else "",
                "average_cloud_percentage": round(sum(clouds) / len(clouds), 4) if clouds else "",
                "number_of_temporal_pairs": pairs,
                "locations_with_insufficient_images": insuf,
            }
        )

    all_dates = sorted(r["acquisition_date"] for r in rows if r.get("acquisition_date"))
    all_clouds = []
    for r in rows:
        try:
            all_clouds.append(float(r["cloud_percentage"]))
        except (TypeError, ValueError):
            pass
    total_pairs = sum(max(0, len(v) - 1) for v in by_loc.values())
    loc_failed = [lid for lid, _ in [(loc["location_id"], loc) for loc in config.LOCATIONS] if lid not in by_loc]
    loc_lt8 = [lid for lid, items in by_loc.items() if len(items) < config.TARGET_IMAGES_PER_LOCATION]

    summary = {
        "total_locations": len(config.LOCATIONS),
        "locations_with_images": len(by_loc),
        "total_images": len(rows),
        "total_temporal_pairs": total_pairs,
        "cities_processed": len(by_city),
        "locations_failed": len(loc_failed),
        "locations_with_lt_8_images": len(loc_lt8),
        "earliest_date": all_dates[0] if all_dates else "",
        "latest_date": all_dates[-1] if all_dates else "",
        "average_cloud_percentage": round(sum(all_clouds) / len(all_clouds), 4) if all_clouds else "",
    }
    return city_rows, summary


def write_summary_report(summary: dict, city_rows: list[dict]) -> None:
    lines = [
        "TerraWatch India Multi-Temporal Sentinel-2 Dataset Summary",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        f"Collection: {config.EE_COLLECTION}",
        "",
        f"total_locations: {summary['total_locations']}",
        f"locations_with_images: {summary['locations_with_images']}",
        f"total_images: {summary['total_images']}",
        f"total_temporal_pairs: {summary['total_temporal_pairs']}",
        f"cities_processed: {summary['cities_processed']}",
        f"locations_failed: {summary['locations_failed']}",
        f"locations_with_<8_images: {summary['locations_with_lt_8_images']}",
        f"earliest_date: {summary['earliest_date']}",
        f"latest_date: {summary['latest_date']}",
        f"average_cloud_percentage: {summary['average_cloud_percentage']}",
        "",
        "Per-city:",
    ]
    for c in city_rows:
        lines.append(
            f"  {c['city']}: images={c['number_of_images']} "
            f"locs={c['number_of_locations']} "
            f"range={c['earliest_date']}→{c['latest_date']} "
            f"avg_cloud={c['average_cloud_percentage']}"
        )
    config.SUMMARY_REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def enrich_rows_from_disk(rows: list[dict]) -> list[dict]:
    """Refresh raster geometry fields from on-disk GeoTIFFs when present."""
    enriched = []
    for row in rows:
        path = config.PACKAGE_DIR / row["image_path"]
        if path.exists():
            try:
                meta = read_geotiff_meta(path)
                row = dict(row)
                row["width"] = meta["width"]
                row["height"] = meta["height"]
                row["crs"] = meta["crs"]
                row["transform"] = str(meta["transform"])
                row["bbox"] = str(meta["bbox"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not read %s: %s", path, exc)
        enriched.append(row)
    return enriched


def scan_images_without_metadata(rows: list[dict]) -> list[Path]:
    known = {str((config.PACKAGE_DIR / r["image_path"]).resolve()) for r in rows if r.get("image_path")}
    orphans = []
    if not config.IMAGES_DIR.exists():
        return orphans
    for path in config.IMAGES_DIR.rglob("*.tif"):
        if str(path.resolve()) not in known:
            orphans.append(path)
    return orphans


def regenerate_all() -> dict:
    setup_logging()
    config.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    write_locations_csv()

    rows = load_csv(config.TEMPORAL_METADATA_CSV)
    rows = enrich_rows_from_disk(rows)
    # Drop rows whose files vanished
    rows = [r for r in rows if (config.PACKAGE_DIR / r["image_path"]).exists()]
    write_csv(
        config.TEMPORAL_METADATA_CSV,
        sorted(rows, key=lambda r: (r["location_id"], r["acquisition_date"])),
        TEMPORAL_FIELDS,
    )

    city_rows, summary = build_statistics(rows)
    write_csv(
        config.DATASET_STATISTICS_CSV,
        city_rows
        + [
            {
                "city": "__SUMMARY__",
                "number_of_locations": summary["locations_with_images"],
                "number_of_images": summary["total_images"],
                "earliest_date": summary["earliest_date"],
                "latest_date": summary["latest_date"],
                "average_cloud_percentage": summary["average_cloud_percentage"],
                "number_of_temporal_pairs": summary["total_temporal_pairs"],
                "locations_with_insufficient_images": summary["locations_with_lt_8_images"],
            }
        ],
        [
            "city",
            "number_of_locations",
            "number_of_images",
            "earliest_date",
            "latest_date",
            "average_cloud_percentage",
            "number_of_temporal_pairs",
            "locations_with_insufficient_images",
        ],
    )
    write_summary_report(summary, city_rows)
    orphans = scan_images_without_metadata(rows)
    if orphans:
        logger.warning("%s GeoTIFF(s) on disk without metadata rows", len(orphans))
    logger.info(
        "Metadata regenerated: %s images, %s temporal pairs",
        summary["total_images"],
        summary["total_temporal_pairs"],
    )
    return summary
