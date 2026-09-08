"""
TerraWatch AI - Temporal Dataset Downloader
Downloads multi-temporal real Sentinel-2 L2A satellite imagery from
Microsoft Planetary Computer STAC API for historical change analysis.

Outputs:
- data/temporal/images/LOC_XXX_YYYY.jpg
- data/temporal/temporal_metadata.csv
"""

from datetime import date, timedelta
from pathlib import Path
import csv
import io
import sys
import time

from PIL import Image
import planetary_computer as pc
import pystac_client
import requests

# ============================================================
# SAFETY GATE: TEST MODE
# ============================================================
# When True: Processes ONLY LOC_001 and 2023 (1 image).
# Set to False ONLY after explicit approval to process all 20 images.
TEST_MODE = True

# ============================================================
# PROJECT PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPORAL_DIR = BASE_DIR / "data" / "temporal"
IMAGES_DIR = TEMPORAL_DIR / "images"
METADATA_FILE = TEMPORAL_DIR / "temporal_metadata.csv"

# ============================================================
# STAC & API CONFIGURATION
# ============================================================
STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DATA_API_BBOX_URL = "https://planetarycomputer.microsoft.com/api/data/v1/item/bbox/{minx},{miny},{maxx},{maxy}/256x256.jpg"

COLLECTION = "sentinel-2-l2a"
MAX_CLOUD_COVER = 30.0
SEARCH_WINDOW_DAYS = 45
TARGET_MONTH = 6
TARGET_DAY = 15

# Half-extent for ~3km x 3km box around coordinate (~0.015 deg lat/lon)
BBOX_HALF_DELTA = 0.015

# ============================================================
# TEMPORAL LOCATIONS & YEARS
# ============================================================
ALL_LOCATIONS = {
    "LOC_001": {"latitude": 51.101323, "longitude": -113.917243},
    "LOC_002": {"latitude": 50.966686, "longitude": -113.917264},
    "LOC_003": {"latitude": 51.051057, "longitude": -113.917465},
    "LOC_004": {"latitude": 51.071225, "longitude": -113.917555},
    "LOC_005": {"latitude": 51.165343, "longitude": -113.917557},
}

ALL_YEARS = [2023, 2024, 2025, 2026]

CSV_COLUMNS = [
    "image_id",
    "location_id",
    "scene_id",
    "latitude",
    "longitude",
    "target_year",
    "acquisition_date",
    "acquisition_datetime",
    "cloud_cover",
    "image_path",
]


def load_existing_metadata() -> tuple[list[dict], set[tuple[str, int]], int]:
    """
    Load existing temporal_metadata.csv to support resumability.
    Returns:
        (rows_list, set of (location_id, target_year), next_image_id)
    """
    rows = []
    processed_keys = set()
    max_id = 0

    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
                    loc_id = row.get("location_id", "")
                    try:
                        yr = int(row.get("target_year", 0))
                        processed_keys.add((loc_id, yr))
                    except ValueError:
                        pass
                    try:
                        img_id = int(row.get("image_id", 0))
                        if img_id > max_id:
                            max_id = img_id
                    except ValueError:
                        pass
        except Exception as e:
            print(f"Warning: Could not read existing metadata file: {e}")

    return rows, processed_keys, max_id + 1


def search_best_scene(catalog, lat: float, lon: float, year: int):
    """
    Search Planetary Computer for the best Sentinel-2 L2A scene
    near June 15 +/- 45 days with cloud cover <= 30%.
    """
    target_dt = date(year, TARGET_MONTH, TARGET_DAY)
    start_dt = target_dt - timedelta(days=SEARCH_WINDOW_DAYS)
    end_dt = target_dt + timedelta(days=SEARCH_WINDOW_DAYS)

    date_range = f"{start_dt.isoformat()}/{end_dt.isoformat()}"

    minx = round(lon - BBOX_HALF_DELTA, 6)
    miny = round(lat - BBOX_HALF_DELTA, 6)
    maxx = round(lon + BBOX_HALF_DELTA, 6)
    maxy = round(lat + BBOX_HALF_DELTA, 6)
    bbox = [minx, miny, maxx, maxy]

    try:
        search = catalog.search(
            collections=[COLLECTION],
            bbox=bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lte": MAX_CLOUD_COVER}},
        )
        items = list(search.items())
    except Exception as e:
        print(f"  [ERROR] STAC search request failed: {e}")
        return None, bbox

    if not items:
        # Fallback: search without strict cloud cover filter if none found under 30%
        try:
            search = catalog.search(
                collections=[COLLECTION],
                bbox=bbox,
                datetime=date_range,
            )
            items = list(search.items())
        except Exception:
            items = []

    if not items:
        return None, bbox

    # Sort items by: 1. Cloud cover, 2. Distance in days from target date (June 15)
    def scene_score(item):
        cc = float(item.properties.get("eo:cloud_cover", 100.0))
        item_date = item.datetime.date() if item.datetime else target_dt
        day_diff = abs((item_date - target_dt).days)
        return (cc, day_diff)

    best_item = min(items, key=scene_score)
    return best_item, bbox


def download_chip(item, bbox: list[float], output_path: Path) -> bool:
    """
    Download a 256x256 RGB JPEG image chip centered around the bounding box.
    """
    minx, miny, maxx, maxy = bbox
    url = (
        f"{DATA_API_BBOX_URL.format(minx=minx, miny=miny, maxx=maxx, maxy=maxy)}"
        f"?collection={COLLECTION}&item={item.id}&assets=visual"
    )

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            print(f"  [ERROR] Download request failed with status {response.status_code}: {response.text[:100]}")
            return False

        content = response.content
        if len(content) < 100:
            print("  [ERROR] Downloaded image content is empty or truncated.")
            return False

        # Validate with Pillow
        img = Image.open(io.BytesIO(content))
        img = img.convert("RGB")
        img.save(output_path, "JPEG", quality=95)

        # Confirm file on disk
        with Image.open(output_path) as verified:
            verified.verify()

        return True

    except Exception as e:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        print(f"  [ERROR] Failed to save/validate image: {e}")
        return False


def save_metadata(rows: list[dict]) -> None:
    """
    Save complete metadata list to data/temporal/temporal_metadata.csv.
    """
    with open(METADATA_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("=" * 70)
    print("TERRAWATCH AI - TEMPORAL DATASET DOWNLOADER")
    print("=" * 70)

    # Ensure directories exist
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Configure active locations and years based on TEST_MODE
    if TEST_MODE:
        print("\n*** TEST MODE ENABLED ***")
        print("Scope: LOC_001 only, Year 2023")
        print("Expected downloads: 1\n")
        locations = {"LOC_001": ALL_LOCATIONS["LOC_001"]}
        years = [2023]
    else:
        print("\n*** FULL PRODUCTION MODE ***")
        print(f"Scope: {len(ALL_LOCATIONS)} locations x {len(ALL_YEARS)} years")
        print(f"Expected downloads: {len(ALL_LOCATIONS) * len(ALL_YEARS)}\n")
        locations = ALL_LOCATIONS
        years = ALL_YEARS

    # Connect to Planetary Computer STAC
    print("Connecting to Microsoft Planetary Computer STAC API...")
    try:
        catalog = pystac_client.Client.open(
            STAC_API_URL,
            modifier=pc.sign_inplace,
        )
        print("Connected successfully!\n")
    except Exception as e:
        print(f"ERROR: Could not connect to STAC API: {e}")
        sys.exit(1)

    # Load existing metadata for resumability
    metadata_rows, processed_pairs, next_id = load_existing_metadata()
    print(f"Existing records in metadata: {len(metadata_rows)}")

    downloaded_count = 0
    skipped_count = 0
    failed_count = 0

    for loc_id, coords in locations.items():
        lat = coords["latitude"]
        lon = coords["longitude"]

        for year in years:
            pair = (loc_id, year)
            filename = f"{loc_id}_{year}.jpg"
            image_dest = IMAGES_DIR / filename
            rel_image_path = f"data/temporal/images/{filename}"

            print("-" * 70)
            print(f"Processing: {loc_id} | Year: {year} | Coords: ({lat:.6f}, {lon:.6f})")

            # Check if already processed
            if pair in processed_pairs and image_dest.exists():
                print(f"  [SKIPPED] {filename} already exists and is recorded in metadata.")
                skipped_count += 1
                continue

            # Search STAC for best scene
            print(f"  Searching real Sentinel-2 L2A acquisitions around June 15, {year}...")
            best_item, bbox = search_best_scene(catalog, lat, lon, year)

            if best_item is None:
                print(f"  [WARNING] No suitable Sentinel-2 scene found for {loc_id} in {year}. Skipping.")
                failed_count += 1
                continue

            acq_dt = best_item.datetime.isoformat() if best_item.datetime else "UNKNOWN"
            acq_date = best_item.datetime.strftime("%Y-%m-%d") if best_item.datetime else "UNKNOWN"
            cloud_cover = round(float(best_item.properties.get("eo:cloud_cover", 0.0)), 2)

            print(f"  Selected Scene ID : {best_item.id}")
            print(f"  Real Acq Date     : {acq_date} ({acq_dt})")
            print(f"  Cloud Cover       : {cloud_cover}%")

            # Download RGB image chip (256x256)
            print(f"  Downloading RGB image chip (256x256) -> {filename}...")
            success = download_chip(best_item, bbox, image_dest)

            if not success:
                print(f"  [FAILED] Could not download image chip for {filename}.")
                failed_count += 1
                continue

            # Record metadata
            new_record = {
                "image_id": next_id,
                "location_id": loc_id,
                "scene_id": best_item.id,
                "latitude": lat,
                "longitude": lon,
                "target_year": year,
                "acquisition_date": acq_date,
                "acquisition_datetime": acq_dt,
                "cloud_cover": cloud_cover,
                "image_path": rel_image_path,
            }

            metadata_rows.append(new_record)
            processed_pairs.add(pair)
            save_metadata(metadata_rows)
            next_id += 1
            downloaded_count += 1

            print(f"  [SUCCESS] Saved image and updated metadata.")

    # Final summary
    print("\n" + "=" * 70)
    print("TEMPORAL DATASET BATCH FINISHED")
    print("=" * 70)
    print(f"Mode                 : {'TEST MODE' if TEST_MODE else 'FULL PRODUCTION'}")
    print(f"Images downloaded    : {downloaded_count}")
    print(f"Images skipped       : {skipped_count}")
    print(f"Failed / Not found   : {failed_count}")
    print(f"Total metadata rows  : {len(metadata_rows)}")
    print(f"Metadata file        : {METADATA_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
