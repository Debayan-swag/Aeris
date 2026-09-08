"""Quick Earth Engine auth + collection smoke test (no downloads)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

from india_multitemporal_sentinel2 import config  # noqa: E402
from india_multitemporal_sentinel2.ee_client import initialize_earth_engine  # noqa: E402
from india_multitemporal_sentinel2.pipeline_utils import fixed_aoi_geojson, logger, setup_logging  # noqa: E402


def main() -> int:
    setup_logging()
    initialize_earth_engine(config.EE_PROJECT)
    import ee

    loc = next(l for l in config.LOCATIONS if l["location_id"] == "DEL_001")
    aoi = fixed_aoi_geojson(loc["longitude"], loc["latitude"], config.AOI_WIDTH_M, config.AOI_HEIGHT_M)
    geom = ee.Geometry.Polygon(aoi["coordinates"])
    col = (
        ee.ImageCollection(config.EE_COLLECTION)
        .filterBounds(geom)
        .filterDate(config.START_DATE, config.END_DATE)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", config.PRIMARY_CLOUD_LIMIT))
    )
    n = col.size().getInfo()
    sample = col.sort("system:time_start").first()
    props = sample.toDictionary(
        ["PRODUCT_ID", "GRANULE_ID", "MGRS_TILE", "CLOUDY_PIXEL_PERCENTAGE", "system:time_start"]
    ).getInfo()
    logger.info("Collection %s | DEL_001 scenes cloud<=%s: %s", config.EE_COLLECTION, config.PRIMARY_CLOUD_LIMIT, n)
    logger.info("Sample EE metadata: %s", props)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
