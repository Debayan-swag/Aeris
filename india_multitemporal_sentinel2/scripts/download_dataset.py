"""
Download real Sentinel-2 SR Harmonized GeoTIFFs for fixed Indian AOIs.

Usage:
  set EE_PROJECT=your-gcp-project
  python scripts/download_dataset.py
  set TW_S2_LOCATION_FILTER=DEL_001
  set TW_S2_MAX_IMAGES=3
  python scripts/download_dataset.py
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import requests

# Allow running as `python scripts/download_dataset.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from india_multitemporal_sentinel2 import config  # noqa: E402
from india_multitemporal_sentinel2.ee_client import initialize_earth_engine  # noqa: E402
from india_multitemporal_sentinel2.pipeline_utils import (  # noqa: E402
    aoi_bounds_wgs84,
    append_csv_row,
    fixed_aoi_geojson,
    load_csv,
    logger,
    ms_to_iso,
    quality_flag,
    read_geotiff_meta,
    retry_call,
    select_temporal_scenes,
    setup_logging,
    write_csv,
)

LOG_FIELDS = [
    "location_id",
    "city",
    "attempted_scene",
    "acquisition_date",
    "cloud_percentage",
    "product_id",
    "status",
    "error_message",
]

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


def log_acquisition(**kwargs) -> None:
    append_csv_row(config.ACQUISITION_LOG_CSV, kwargs, LOG_FIELDS)


def image_filename(location_id: str, acquisition_date: str) -> str:
    return f"{location_id}_{acquisition_date}.tif"


def image_output_path(city: str, location_id: str, acquisition_date: str) -> Path:
    return config.IMAGES_DIR / city / location_id / image_filename(location_id, acquisition_date)


def verify_existing_geotiff(path: Path) -> dict | None:
    try:
        meta = read_geotiff_meta(path)
        if meta["count"] < 6:
            return None
        return meta
    except Exception:  # noqa: BLE001
        return None


def list_scenes_for_aoi(ee, aoi_geom, cloud_limit: float) -> list[dict]:
    """Query EE for real scene metadata intersecting the fixed AOI."""
    collection = (
        ee.ImageCollection(config.EE_COLLECTION)
        .filterBounds(aoi_geom)
        .filterDate(config.START_DATE, config.END_DATE)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_limit))
        .sort("system:time_start")
    )

    def _fetch():
        return {
            "system_index": collection.aggregate_array("system:index").getInfo(),
            "time_start": collection.aggregate_array("system:time_start").getInfo(),
            "cloud": collection.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo(),
            "product_id": collection.aggregate_array("PRODUCT_ID").getInfo(),
            "granule_id": collection.aggregate_array("GRANULE_ID").getInfo(),
            "mgrs_tile": collection.aggregate_array("MGRS_TILE").getInfo(),
            "spacecraft": collection.aggregate_array("SPACECRAFT_NAME").getInfo(),
        }

    raw = retry_call(
        _fetch,
        retries=config.MAX_RETRIES,
        base_delay=config.RETRY_BASE_DELAY_SEC,
        what=f"EE scene list cloud<={cloud_limit}",
    )

    scenes: list[dict] = []
    n = len(raw["system_index"] or [])
    for i in range(n):
        ts = raw["time_start"][i]
        cloud = raw["cloud"][i]
        if ts is None or cloud is None:
            continue
        acq_date, acq_dt, year, month = ms_to_iso(ts)
        scenes.append(
            {
                "system_index": raw["system_index"][i],
                "acquisition_date": acq_date,
                "acquisition_datetime": acq_dt,
                "year": year,
                "month": month,
                "cloud_percentage": float(cloud),
                "product_id": (raw["product_id"][i] if raw["product_id"] else None) or "",
                "granule_id": (raw["granule_id"][i] if raw["granule_id"] else None) or "",
                "mgrs_tile": (raw["mgrs_tile"][i] if raw["mgrs_tile"] else None) or "",
                "spacecraft_name": (raw["spacecraft"][i] if raw["spacecraft"] else None)
                or config.SENSOR_NAME,
                "time_start_ms": int(ts),
            }
        )
    # Deduplicate by acquisition_date (keep lowest cloud)
    by_date: dict[str, dict] = {}
    for s in scenes:
        d = s["acquisition_date"]
        if d not in by_date or s["cloud_percentage"] < by_date[d]["cloud_percentage"]:
            by_date[d] = s
    return sorted(by_date.values(), key=lambda s: s["acquisition_date"])


def build_export_image(ee, system_index: str, aoi_geom):
    img = ee.Image(f"{config.EE_COLLECTION}/{system_index}").select(config.BANDS)
    if config.ENABLE_PIXEL_CLOUD_MASK:
        # Optional pixel-level mask; does not change acquisition date / scene cloud %
        s2c = (
            ee.ImageCollection(config.EE_CLOUD_PROBABILITY_COLLECTION)
            .filter(ee.Filter.eq("system:index", system_index))
            .first()
        )

        def _apply_mask(probability_img):
            prob = ee.Image(probability_img).select("probability")
            return img.updateMask(prob.lt(config.CLOUD_PROBABILITY_THRESHOLD))

        img = ee.Algorithms.If(s2c, _apply_mask(s2c), img)
        img = ee.Image(img)
    return img.clip(aoi_geom)


def download_geotiff(ee, image, aoi_geom, out_path: Path, utm_epsg: int) -> None:
    """Download a GeoTIFF for the fixed AOI via getDownloadURL (real EE pixels)."""
    # Fixed region + UTM CRS + dimensions ⇒ same footprint/size across dates (~10 m).
    params = {
        "region": aoi_geom,
        "crs": f"EPSG:{utm_epsg}",
        "dimensions": [config.PATCH_SIZE, config.PATCH_SIZE],
        "format": "GEO_TIFF",
        "bands": config.BANDS,
        "filePerBand": False,
    }
    url = image.getDownloadURL(params)

    def _fetch():
        resp = requests.get(url, timeout=config.DOWNLOAD_TIMEOUT_SEC)
        resp.raise_for_status()
        return resp.content

    content = retry_call(
        _fetch,
        retries=config.MAX_RETRIES,
        base_delay=config.RETRY_BASE_DELAY_SEC,
        what=f"download {out_path.name}",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # getDownloadURL may return a zip containing a GeoTIFF or raw GeoTIFF bytes
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff"))]
            if not names:
                raise RuntimeError(f"Zip from EE contained no GeoTIFF: {zf.namelist()}")
            out_path.write_bytes(zf.read(names[0]))
    else:
        out_path.write_bytes(content)


def make_metadata_row(loc: dict, scene: dict, rel_path: str, raster_meta: dict) -> dict:
    cloud = float(scene["cloud_percentage"])
    return {
        "image_id": f"{loc['location_id']}_{scene['acquisition_date'].replace('-', '')}",
        "location_id": loc["location_id"],
        "city": loc["city"],
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "acquisition_date": scene["acquisition_date"],
        "acquisition_datetime": scene["acquisition_datetime"],
        "year": scene["year"],
        "month": scene["month"],
        "sensor": scene.get("spacecraft_name") or config.SENSOR_NAME,
        "collection": config.EE_COLLECTION,
        "product_id": scene.get("product_id") or "",
        "granule_id": scene.get("granule_id") or "",
        "mgrs_tile": scene.get("mgrs_tile") or "",
        "cloud_percentage": cloud,
        "image_path": rel_path.replace("\\", "/"),
        "bands": "|".join(config.BANDS),
        "resolution": config.RESOLUTION,
        "width": raster_meta["width"],
        "height": raster_meta["height"],
        "crs": raster_meta["crs"],
        "transform": str(raster_meta["transform"]),
        "bbox": str(raster_meta["bbox"]),
        "quality_flag": quality_flag(cloud, config.PRIMARY_CLOUD_LIMIT, config.FALLBACK_CLOUD_LIMIT),
    }


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


def upsert_temporal_row(row: dict) -> None:
    existing = load_csv(config.TEMPORAL_METADATA_CSV)
    by_id = {r["image_id"]: r for r in existing}
    by_id[row["image_id"]] = row
    write_csv(
        config.TEMPORAL_METADATA_CSV,
        sorted(by_id.values(), key=lambda r: (r["location_id"], r["acquisition_date"])),
        TEMPORAL_FIELDS,
    )


def process_location(ee, loc: dict, target_n: int) -> dict:
    location_id = loc["location_id"]
    city = loc["city"]
    logger.info("=== %s (%s) ===", location_id, city)

    aoi_gj = fixed_aoi_geojson(
        loc["longitude"], loc["latitude"], config.AOI_WIDTH_M, config.AOI_HEIGHT_M
    )
    coords = aoi_gj["coordinates"]
    aoi_geom = ee.Geometry.Polygon(coords)

    scenes = list_scenes_for_aoi(ee, aoi_geom, config.PRIMARY_CLOUD_LIMIT)
    used_fallback = False
    if len(scenes) < target_n:
        logger.info(
            "%s: only %s scenes at cloud<=%s; trying fallback <=%s",
            location_id,
            len(scenes),
            config.PRIMARY_CLOUD_LIMIT,
            config.FALLBACK_CLOUD_LIMIT,
        )
        scenes = list_scenes_for_aoi(ee, aoi_geom, config.FALLBACK_CLOUD_LIMIT)
        used_fallback = True

    selected = select_temporal_scenes(scenes, target_n)
    if not selected:
        log_acquisition(
            location_id=location_id,
            city=city,
            attempted_scene="",
            acquisition_date="",
            cloud_percentage="",
            product_id="",
            status="insufficient_temporal_coverage",
            error_message=f"No scenes with cloud <= {config.FALLBACK_CLOUD_LIMIT}",
        )
        return {"location_id": location_id, "ok": False, "downloaded": 0, "existing": 0}

    downloaded = 0
    existing = 0
    failed = 0

    for scene in selected:
        acq = scene["acquisition_date"]
        out_path = image_output_path(city, location_id, acq)
        rel = str(out_path.relative_to(config.PACKAGE_DIR))

        if out_path.exists():
            meta = verify_existing_geotiff(out_path)
            if meta:
                row = make_metadata_row(loc, scene, rel, meta)
                upsert_temporal_row(row)
                log_acquisition(
                    location_id=location_id,
                    city=city,
                    attempted_scene=scene.get("system_index") or "",
                    acquisition_date=acq,
                    cloud_percentage=scene["cloud_percentage"],
                    product_id=scene.get("product_id") or "",
                    status="already_exists",
                    error_message="",
                )
                existing += 1
                continue
            logger.warning("Existing file invalid, re-downloading: %s", out_path)

        try:
            image = build_export_image(ee, scene["system_index"], aoi_geom)
            download_geotiff(ee, image, aoi_geom, out_path, aoi_gj["crs_epsg"])
            meta = verify_existing_geotiff(out_path)
            if not meta:
                raise RuntimeError("Downloaded GeoTIFF failed verification")
            row = make_metadata_row(loc, scene, rel, meta)
            upsert_temporal_row(row)
            log_acquisition(
                location_id=location_id,
                city=city,
                attempted_scene=scene.get("system_index") or "",
                acquisition_date=acq,
                cloud_percentage=scene["cloud_percentage"],
                product_id=scene.get("product_id") or "",
                status="downloaded",
                error_message="fallback_cloud" if used_fallback and scene["cloud_percentage"] > config.PRIMARY_CLOUD_LIMIT else "",
            )
            downloaded += 1
            logger.info(
                "Downloaded %s cloud=%.2f product=%s",
                out_path.name,
                scene["cloud_percentage"],
                scene.get("product_id"),
            )
        except Exception as exc:  # noqa: BLE001 - do not abort whole dataset
            failed += 1
            if out_path.exists():
                try:
                    out_path.unlink()
                except OSError:
                    pass
            log_acquisition(
                location_id=location_id,
                city=city,
                attempted_scene=scene.get("system_index") or "",
                acquisition_date=acq,
                cloud_percentage=scene["cloud_percentage"],
                product_id=scene.get("product_id") or "",
                status="failed",
                error_message=str(exc)[:500],
            )
            logger.error("%s %s failed: %s", location_id, acq, exc)

    ok = (downloaded + existing) > 0
    if not ok:
        log_acquisition(
            location_id=location_id,
            city=city,
            attempted_scene="",
            acquisition_date="",
            cloud_percentage="",
            product_id="",
            status="insufficient_temporal_coverage",
            error_message="All selected scene downloads failed",
        )
    return {
        "location_id": location_id,
        "ok": ok,
        "downloaded": downloaded,
        "existing": existing,
        "failed": failed,
        "selected": len(selected),
        "available": len(scenes),
    }


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Download Indian multi-temporal Sentinel-2 dataset")
    parser.add_argument("--project", default=config.EE_PROJECT, help="Earth Engine / GCP project")
    parser.add_argument(
        "--locations",
        default="",
        help="Comma-separated location_ids (overrides TW_S2_LOCATION_FILTER)",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Override target images/location")
    args = parser.parse_args()

    if args.locations:
        config.LOCATION_FILTER[:] = [x.strip() for x in args.locations.split(",") if x.strip()]
    target_n = args.max_images or config.target_images_per_location()

    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    write_locations_csv()

    initialize_earth_engine(args.project)
    import ee

    locations = config.get_locations()
    logger.info(
        "Processing %s locations | target_images=%s | dates=%s→%s | collection=%s",
        len(locations),
        target_n,
        config.START_DATE,
        config.END_DATE,
        config.EE_COLLECTION,
    )

    results = []
    for loc in locations:
        try:
            results.append(process_location(ee, loc, target_n))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Location %s aborted: %s", loc["location_id"], exc)
            log_acquisition(
                location_id=loc["location_id"],
                city=loc["city"],
                attempted_scene="",
                acquisition_date="",
                cloud_percentage="",
                product_id="",
                status="failed",
                error_message=str(exc)[:500],
            )
            results.append(
                {"location_id": loc["location_id"], "ok": False, "downloaded": 0, "existing": 0}
            )

    from india_multitemporal_sentinel2.metadata_builder import regenerate_all

    regenerate_all()

    ok_n = sum(1 for r in results if r.get("ok"))
    fail_n = len(results) - ok_n
    total_new = sum(r.get("downloaded", 0) for r in results)
    total_exist = sum(r.get("existing", 0) for r in results)
    logger.info(
        "DONE locations_ok=%s failed=%s newly_downloaded=%s already_existed=%s",
        ok_n,
        fail_n,
        total_new,
        total_exist,
    )
    for r in results:
        mark = "OK" if r.get("ok") else "FAIL"
        logger.info(
            "  %s %s  new=%s exist=%s",
            mark,
            r["location_id"],
            r.get("downloaded", 0),
            r.get("existing", 0),
        )
    return 0 if ok_n else 1


if __name__ == "__main__":
    raise SystemExit(main())
