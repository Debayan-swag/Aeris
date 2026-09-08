"""
Configuration for the Indian Multi-Temporal Sentinel-2 acquisition pipeline.

All acquisition dates, cloud percentages, and product IDs come from Earth Engine
at runtime — nothing in this file fabricates scene metadata.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = PACKAGE_DIR / "images"
METADATA_DIR = PACKAGE_DIR / "metadata"
SCRIPTS_DIR = PACKAGE_DIR / "scripts"

LOCATIONS_CSV = METADATA_DIR / "locations.csv"
TEMPORAL_METADATA_CSV = METADATA_DIR / "temporal_metadata.csv"
DATASET_STATISTICS_CSV = METADATA_DIR / "dataset_statistics.csv"
ACQUISITION_LOG_CSV = METADATA_DIR / "acquisition_log.csv"
CHANGE_CANDIDATES_CSV = METADATA_DIR / "change_candidates.csv"
SUMMARY_REPORT_TXT = METADATA_DIR / "dataset_summary.txt"
VALIDATION_REPORT_TXT = METADATA_DIR / "validation_report.txt"

# ---------------------------------------------------------------------------
# Earth Engine
# ---------------------------------------------------------------------------
EE_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
EE_CLOUD_PROBABILITY_COLLECTION = "COPERNICUS/S2_CLOUD_PROBABILITY"
EE_PROJECT = os.environ.get("EE_PROJECT", "").strip() or None

BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
SENSOR_NAME = "Sentinel-2"

# ---------------------------------------------------------------------------
# Spatial / temporal acquisition parameters
# ---------------------------------------------------------------------------
START_DATE = os.environ.get("TW_S2_START_DATE", "2018-01-01")
END_DATE = os.environ.get("TW_S2_END_DATE", date.today().isoformat())

PRIMARY_CLOUD_LIMIT = float(os.environ.get("TW_S2_PRIMARY_CLOUD", "10"))
FALLBACK_CLOUD_LIMIT = float(os.environ.get("TW_S2_FALLBACK_CLOUD", "20"))
TARGET_IMAGES_PER_LOCATION = int(os.environ.get("TW_S2_TARGET_IMAGES", "8"))

PATCH_SIZE = int(os.environ.get("TW_S2_PATCH_SIZE", "256"))  # pixels
RESOLUTION = float(os.environ.get("TW_S2_RESOLUTION", "10"))  # meters
AOI_WIDTH_M = PATCH_SIZE * RESOLUTION
AOI_HEIGHT_M = PATCH_SIZE * RESOLUTION

# Optional pixel-level cloud masking (scene-level CLOUDY_PIXEL_PERCENTAGE always kept)
ENABLE_PIXEL_CLOUD_MASK = os.environ.get("TW_S2_PIXEL_CLOUD_MASK", "0").lower() in {
    "1",
    "true",
    "yes",
}
CLOUD_PROBABILITY_THRESHOLD = float(os.environ.get("TW_S2_CLOUD_PROB_THRESHOLD", "60"))

# ---------------------------------------------------------------------------
# Download robustness
# ---------------------------------------------------------------------------
MAX_RETRIES = int(os.environ.get("TW_S2_MAX_RETRIES", "5"))
RETRY_BASE_DELAY_SEC = float(os.environ.get("TW_S2_RETRY_BASE_DELAY", "2.0"))
DOWNLOAD_TIMEOUT_SEC = float(os.environ.get("TW_S2_DOWNLOAD_TIMEOUT", "300"))

# Test / scoped runs (comma-separated location_ids, empty = all)
LOCATION_FILTER = [
    x.strip()
    for x in os.environ.get("TW_S2_LOCATION_FILTER", "").split(",")
    if x.strip()
]
MAX_IMAGES_OVERRIDE = os.environ.get("TW_S2_MAX_IMAGES")
MAX_IMAGES_OVERRIDE = int(MAX_IMAGES_OVERRIDE) if MAX_IMAGES_OVERRIDE else None

# ---------------------------------------------------------------------------
# Fixed Indian locations (20 cities × 5 sites = 100)
# Coordinates are fixed spatial identities — never changed per date.
# ---------------------------------------------------------------------------
LOCATIONS: list[dict] = [
    # Delhi
    {"location_id": "DEL_001", "city": "Delhi", "latitude": 28.6139, "longitude": 77.2090, "location_type": "urban"},
    {"location_id": "DEL_002", "city": "Delhi", "latitude": 28.7041, "longitude": 77.1025, "location_type": "mixed"},
    {"location_id": "DEL_003", "city": "Delhi", "latitude": 28.5355, "longitude": 77.3910, "location_type": "construction"},
    {"location_id": "DEL_004", "city": "Delhi", "latitude": 28.6692, "longitude": 77.4538, "location_type": "industrial"},
    {"location_id": "DEL_005", "city": "Delhi", "latitude": 28.5562, "longitude": 77.1000, "location_type": "vegetation"},
    # Mumbai
    {"location_id": "MUM_001", "city": "Mumbai", "latitude": 19.0760, "longitude": 72.8777, "location_type": "urban"},
    {"location_id": "MUM_002", "city": "Mumbai", "latitude": 19.2183, "longitude": 72.9781, "location_type": "mixed"},
    {"location_id": "MUM_003", "city": "Mumbai", "latitude": 19.2403, "longitude": 73.1305, "location_type": "construction"},
    {"location_id": "MUM_004", "city": "Mumbai", "latitude": 19.0178, "longitude": 73.0410, "location_type": "industrial"},
    {"location_id": "MUM_005", "city": "Mumbai", "latitude": 19.0607, "longitude": 72.8698, "location_type": "water"},
    # Kolkata
    {"location_id": "KOL_001", "city": "Kolkata", "latitude": 22.5726, "longitude": 88.3639, "location_type": "urban"},
    {"location_id": "KOL_002", "city": "Kolkata", "latitude": 22.5720, "longitude": 88.4370, "location_type": "mixed"},
    {"location_id": "KOL_003", "city": "Kolkata", "latitude": 22.5600, "longitude": 88.5000, "location_type": "construction"},
    {"location_id": "KOL_004", "city": "Kolkata", "latitude": 22.6100, "longitude": 88.4200, "location_type": "water"},
    {"location_id": "KOL_005", "city": "Kolkata", "latitude": 22.5200, "longitude": 88.3300, "location_type": "vegetation"},
    # Bengaluru
    {"location_id": "BLR_001", "city": "Bengaluru", "latitude": 12.9716, "longitude": 77.5946, "location_type": "urban"},
    {"location_id": "BLR_002", "city": "Bengaluru", "latitude": 13.0358, "longitude": 77.5970, "location_type": "mixed"},
    {"location_id": "BLR_003", "city": "Bengaluru", "latitude": 12.8452, "longitude": 77.6602, "location_type": "construction"},
    {"location_id": "BLR_004", "city": "Bengaluru", "latitude": 13.0827, "longitude": 77.5877, "location_type": "industrial"},
    {"location_id": "BLR_005", "city": "Bengaluru", "latitude": 12.9352, "longitude": 77.6245, "location_type": "vegetation"},
    # Chennai
    {"location_id": "CHE_001", "city": "Chennai", "latitude": 13.0827, "longitude": 80.2707, "location_type": "urban"},
    {"location_id": "CHE_002", "city": "Chennai", "latitude": 12.9716, "longitude": 80.2200, "location_type": "mixed"},
    {"location_id": "CHE_003", "city": "Chennai", "latitude": 12.8350, "longitude": 80.2260, "location_type": "construction"},
    {"location_id": "CHE_004", "city": "Chennai", "latitude": 13.1500, "longitude": 80.2900, "location_type": "industrial"},
    {"location_id": "CHE_005", "city": "Chennai", "latitude": 12.9600, "longitude": 80.1700, "location_type": "water"},
    # Hyderabad
    {"location_id": "HYD_001", "city": "Hyderabad", "latitude": 17.3850, "longitude": 78.4867, "location_type": "urban"},
    {"location_id": "HYD_002", "city": "Hyderabad", "latitude": 17.4500, "longitude": 78.3800, "location_type": "mixed"},
    {"location_id": "HYD_003", "city": "Hyderabad", "latitude": 17.4200, "longitude": 78.5500, "location_type": "construction"},
    {"location_id": "HYD_004", "city": "Hyderabad", "latitude": 17.5500, "longitude": 78.4500, "location_type": "industrial"},
    {"location_id": "HYD_005", "city": "Hyderabad", "latitude": 17.3300, "longitude": 78.4200, "location_type": "water"},
    # Ahmedabad
    {"location_id": "AMD_001", "city": "Ahmedabad", "latitude": 23.0225, "longitude": 72.5714, "location_type": "urban"},
    {"location_id": "AMD_002", "city": "Ahmedabad", "latitude": 23.0900, "longitude": 72.5400, "location_type": "mixed"},
    {"location_id": "AMD_003", "city": "Ahmedabad", "latitude": 23.0200, "longitude": 72.6900, "location_type": "construction"},
    {"location_id": "AMD_004", "city": "Ahmedabad", "latitude": 23.0800, "longitude": 72.6500, "location_type": "industrial"},
    {"location_id": "AMD_005", "city": "Ahmedabad", "latitude": 23.0300, "longitude": 72.5000, "location_type": "water"},
    # Pune
    {"location_id": "PUN_001", "city": "Pune", "latitude": 18.5204, "longitude": 73.8567, "location_type": "urban"},
    {"location_id": "PUN_002", "city": "Pune", "latitude": 18.6000, "longitude": 73.7500, "location_type": "mixed"},
    {"location_id": "PUN_003", "city": "Pune", "latitude": 18.4500, "longitude": 73.9500, "location_type": "construction"},
    {"location_id": "PUN_004", "city": "Pune", "latitude": 18.5700, "longitude": 73.9200, "location_type": "industrial"},
    {"location_id": "PUN_005", "city": "Pune", "latitude": 18.4800, "longitude": 73.8000, "location_type": "vegetation"},
    # Jaipur
    {"location_id": "JAI_001", "city": "Jaipur", "latitude": 26.9124, "longitude": 75.7873, "location_type": "urban"},
    {"location_id": "JAI_002", "city": "Jaipur", "latitude": 26.9600, "longitude": 75.8000, "location_type": "mixed"},
    {"location_id": "JAI_003", "city": "Jaipur", "latitude": 26.8200, "longitude": 75.9000, "location_type": "construction"},
    {"location_id": "JAI_004", "city": "Jaipur", "latitude": 26.9500, "longitude": 75.7000, "location_type": "industrial"},
    {"location_id": "JAI_005", "city": "Jaipur", "latitude": 26.8800, "longitude": 75.7500, "location_type": "vegetation"},
    # Lucknow
    {"location_id": "LKO_001", "city": "Lucknow", "latitude": 26.8467, "longitude": 80.9462, "location_type": "urban"},
    {"location_id": "LKO_002", "city": "Lucknow", "latitude": 26.9000, "longitude": 81.0000, "location_type": "mixed"},
    {"location_id": "LKO_003", "city": "Lucknow", "latitude": 26.7600, "longitude": 81.0500, "location_type": "construction"},
    {"location_id": "LKO_004", "city": "Lucknow", "latitude": 26.8700, "longitude": 80.8500, "location_type": "industrial"},
    {"location_id": "LKO_005", "city": "Lucknow", "latitude": 26.8200, "longitude": 80.9000, "location_type": "vegetation"},
    # Surat
    {"location_id": "SRT_001", "city": "Surat", "latitude": 21.1702, "longitude": 72.8311, "location_type": "urban"},
    {"location_id": "SRT_002", "city": "Surat", "latitude": 21.2200, "longitude": 72.9000, "location_type": "mixed"},
    {"location_id": "SRT_003", "city": "Surat", "latitude": 21.1200, "longitude": 72.9500, "location_type": "construction"},
    {"location_id": "SRT_004", "city": "Surat", "latitude": 21.2500, "longitude": 72.8000, "location_type": "industrial"},
    {"location_id": "SRT_005", "city": "Surat", "latitude": 21.1500, "longitude": 72.7500, "location_type": "water"},
    # Kanpur
    {"location_id": "KNP_001", "city": "Kanpur", "latitude": 26.4499, "longitude": 80.3319, "location_type": "urban"},
    {"location_id": "KNP_002", "city": "Kanpur", "latitude": 26.5000, "longitude": 80.4000, "location_type": "mixed"},
    {"location_id": "KNP_003", "city": "Kanpur", "latitude": 26.4000, "longitude": 80.5000, "location_type": "construction"},
    {"location_id": "KNP_004", "city": "Kanpur", "latitude": 26.4800, "longitude": 80.2500, "location_type": "industrial"},
    {"location_id": "KNP_005", "city": "Kanpur", "latitude": 26.3500, "longitude": 80.3000, "location_type": "vegetation"},
    # Nagpur
    {"location_id": "NGP_001", "city": "Nagpur", "latitude": 21.1458, "longitude": 79.0882, "location_type": "urban"},
    {"location_id": "NGP_002", "city": "Nagpur", "latitude": 21.2000, "longitude": 79.1000, "location_type": "mixed"},
    {"location_id": "NGP_003", "city": "Nagpur", "latitude": 21.1000, "longitude": 79.2500, "location_type": "construction"},
    {"location_id": "NGP_004", "city": "Nagpur", "latitude": 21.1800, "longitude": 78.9800, "location_type": "industrial"},
    {"location_id": "NGP_005", "city": "Nagpur", "latitude": 21.0500, "longitude": 79.0500, "location_type": "vegetation"},
    # Indore
    {"location_id": "IDR_001", "city": "Indore", "latitude": 22.7196, "longitude": 75.8577, "location_type": "urban"},
    {"location_id": "IDR_002", "city": "Indore", "latitude": 22.7600, "longitude": 75.9000, "location_type": "mixed"},
    {"location_id": "IDR_003", "city": "Indore", "latitude": 22.6500, "longitude": 76.0000, "location_type": "construction"},
    {"location_id": "IDR_004", "city": "Indore", "latitude": 22.7500, "longitude": 75.7800, "location_type": "industrial"},
    {"location_id": "IDR_005", "city": "Indore", "latitude": 22.6800, "longitude": 75.8200, "location_type": "vegetation"},
    # Bhopal
    {"location_id": "BHO_001", "city": "Bhopal", "latitude": 23.2599, "longitude": 77.4126, "location_type": "urban"},
    {"location_id": "BHO_002", "city": "Bhopal", "latitude": 23.3000, "longitude": 77.4500, "location_type": "mixed"},
    {"location_id": "BHO_003", "city": "Bhopal", "latitude": 23.2000, "longitude": 77.5500, "location_type": "construction"},
    {"location_id": "BHO_004", "city": "Bhopal", "latitude": 23.2800, "longitude": 77.3000, "location_type": "industrial"},
    {"location_id": "BHO_005", "city": "Bhopal", "latitude": 23.2300, "longitude": 77.3500, "location_type": "water"},
    # Patna
    {"location_id": "PAT_001", "city": "Patna", "latitude": 25.5941, "longitude": 85.1376, "location_type": "urban"},
    {"location_id": "PAT_002", "city": "Patna", "latitude": 25.6500, "longitude": 85.1800, "location_type": "mixed"},
    {"location_id": "PAT_003", "city": "Patna", "latitude": 25.5500, "longitude": 85.3000, "location_type": "construction"},
    {"location_id": "PAT_004", "city": "Patna", "latitude": 25.6200, "longitude": 85.0500, "location_type": "industrial"},
    {"location_id": "PAT_005", "city": "Patna", "latitude": 25.5200, "longitude": 85.1000, "location_type": "water"},
    # Vadodara
    {"location_id": "VAD_001", "city": "Vadodara", "latitude": 22.3072, "longitude": 73.1812, "location_type": "urban"},
    {"location_id": "VAD_002", "city": "Vadodara", "latitude": 22.3500, "longitude": 73.2200, "location_type": "mixed"},
    {"location_id": "VAD_003", "city": "Vadodara", "latitude": 22.2500, "longitude": 73.3000, "location_type": "construction"},
    {"location_id": "VAD_004", "city": "Vadodara", "latitude": 22.3200, "longitude": 73.1000, "location_type": "industrial"},
    {"location_id": "VAD_005", "city": "Vadodara", "latitude": 22.2800, "longitude": 73.1500, "location_type": "vegetation"},
    # Coimbatore
    {"location_id": "CBE_001", "city": "Coimbatore", "latitude": 11.0168, "longitude": 76.9558, "location_type": "urban"},
    {"location_id": "CBE_002", "city": "Coimbatore", "latitude": 11.0800, "longitude": 77.0000, "location_type": "mixed"},
    {"location_id": "CBE_003", "city": "Coimbatore", "latitude": 10.9500, "longitude": 77.1000, "location_type": "construction"},
    {"location_id": "CBE_004", "city": "Coimbatore", "latitude": 11.0500, "longitude": 76.9000, "location_type": "industrial"},
    {"location_id": "CBE_005", "city": "Coimbatore", "latitude": 10.9800, "longitude": 76.8500, "location_type": "vegetation"},
    # Kochi
    {"location_id": "KOC_001", "city": "Kochi", "latitude": 9.9312, "longitude": 76.2673, "location_type": "urban"},
    {"location_id": "KOC_002", "city": "Kochi", "latitude": 10.0000, "longitude": 76.3000, "location_type": "mixed"},
    {"location_id": "KOC_003", "city": "Kochi", "latitude": 9.9000, "longitude": 76.4000, "location_type": "construction"},
    {"location_id": "KOC_004", "city": "Kochi", "latitude": 10.0500, "longitude": 76.2500, "location_type": "industrial"},
    {"location_id": "KOC_005", "city": "Kochi", "latitude": 9.9500, "longitude": 76.2000, "location_type": "water"},
    # Bhubaneswar
    {"location_id": "BBS_001", "city": "Bhubaneswar", "latitude": 20.2961, "longitude": 85.8245, "location_type": "urban"},
    {"location_id": "BBS_002", "city": "Bhubaneswar", "latitude": 20.3500, "longitude": 85.8500, "location_type": "mixed"},
    {"location_id": "BBS_003", "city": "Bhubaneswar", "latitude": 20.2500, "longitude": 85.9500, "location_type": "construction"},
    {"location_id": "BBS_004", "city": "Bhubaneswar", "latitude": 20.3000, "longitude": 85.7500, "location_type": "industrial"},
    {"location_id": "BBS_005", "city": "Bhubaneswar", "latitude": 20.2200, "longitude": 85.8000, "location_type": "vegetation"},
]


def get_locations() -> list[dict]:
    """Return configured locations, optionally filtered by TW_S2_LOCATION_FILTER."""
    if not LOCATION_FILTER:
        return list(LOCATIONS)
    wanted = set(LOCATION_FILTER)
    return [loc for loc in LOCATIONS if loc["location_id"] in wanted]


def target_images_per_location() -> int:
    if MAX_IMAGES_OVERRIDE is not None:
        return MAX_IMAGES_OVERRIDE
    return TARGET_IMAGES_PER_LOCATION


def quality_flag_for_cloud(cloud_pct: float) -> str:
    if cloud_pct <= PRIMARY_CLOUD_LIMIT:
        return "good"
    if cloud_pct <= FALLBACK_CLOUD_LIMIT:
        return "acceptable_fallback"
    return "too_cloudy"
