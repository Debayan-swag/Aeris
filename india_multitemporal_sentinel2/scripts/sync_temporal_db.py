"""
Create temporal_images table in database/terrawatch.db if desired.

Does not modify images / image_embeddings / change_events / aoi_monitors.
Does not rebuild FAISS.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

from india_multitemporal_sentinel2 import config  # noqa: E402
from india_multitemporal_sentinel2.pipeline_utils import load_csv, logger, setup_logging  # noqa: E402

DB_PATH = ROOT.parent / "database" / "terrawatch.db"


DDL = """
CREATE TABLE IF NOT EXISTS temporal_images (
    image_id TEXT PRIMARY KEY,
    location_id TEXT NOT NULL,
    city TEXT,
    acquisition_date TEXT,
    image_path TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    cloud_percentage REAL,
    sensor TEXT,
    product_id TEXT,
    collection TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def main() -> int:
    setup_logging()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(DDL)
    rows = load_csv(config.TEMPORAL_METADATA_CSV)
    upsert = """
    INSERT INTO temporal_images (
        image_id, location_id, city, acquisition_date, image_path,
        latitude, longitude, cloud_percentage, sensor, product_id, collection
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(image_id) DO UPDATE SET
        image_path=excluded.image_path,
        cloud_percentage=excluded.cloud_percentage,
        product_id=excluded.product_id
    """
    for r in rows:
        conn.execute(
            upsert,
            (
                r["image_id"],
                r["location_id"],
                r["city"],
                r["acquisition_date"],
                r["image_path"],
                float(r["latitude"]),
                float(r["longitude"]),
                float(r["cloud_percentage"]) if r.get("cloud_percentage") != "" else None,
                r.get("sensor"),
                r.get("product_id"),
                r.get("collection"),
            ),
        )
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM temporal_images").fetchone()[0]
    conn.close()
    logger.info("temporal_images synced: %s rows in %s", n, DB_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
