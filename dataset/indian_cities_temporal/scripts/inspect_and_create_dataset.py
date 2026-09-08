"""Build a conservative, local-only Indian multi-temporal image CSV.

This utility never downloads data or calls external services.  A record is only
selected when the same source row explicitly supplies a city and acquisition
date.  Coordinates alone, and unlinked city catalogues, are deliberately not
used to name an image.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image

REPO = Path(__file__).resolve().parents[3]
OUTPUT = REPO / "dataset" / "indian_cities_temporal"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
EXCLUDED_PARTS = {".git", ".venv", "node_modules", "__pycache__", "indian_cities_temporal"}
CITY_KEYS = ("city", "city_name", "municipality")
DATE_KEYS = ("acquisition_date", "acquisition_datetime", "timestamp", "datetime", "date")
PATH_KEYS = ("image_path", "path", "file_path", "image_file", "filename")
LOCATION_KEYS = ("location_id", "scene_id", "location", "site_id", "tile_id")


def normal(value: Any) -> str:
    return "" if value is None else str(value).strip()


def find_key(row: dict[str, Any], choices: tuple[str, ...]) -> str | None:
    by_lower = {str(k).lower(): k for k in row}
    return next((by_lower[key] for key in choices if key in by_lower), None)


def parse_date(value: Any) -> str | None:
    text = normal(value)
    if not text:
        return None
    # Metadata ISO date or ISO timestamp only; no guessed formats.
    match = re.match(r"^(\d{4}-\d{2}-\d{2})(?:[T\s].*)?$", text)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1)).isoformat()
    except ValueError:
        return None


def resolve_path(value: str, metadata_file: Path) -> Path | None:
    value = value.replace("\\", "/")
    raw = Path(value)
    attempts = [raw if raw.is_absolute() else REPO / raw, metadata_file.parent / raw]
    for attempt in attempts:
        if attempt.is_file():
            return attempt.resolve()
    return None


def repository_files() -> list[Path]:
    return [p for p in REPO.rglob("*") if p.is_file() and not any(x in p.parts for x in EXCLUDED_PARTS)]


def indian_city_catalog(csv_files: list[Path]) -> set[str]:
    """Use only existing city rows whose own coordinates are within India."""
    cities: set[str] = set()
    for source in csv_files:
        try:
            with source.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    city_key = find_key(row, CITY_KEYS)
                    lat_key, lon_key = find_key(row, ("latitude", "lat")), find_key(row, ("longitude", "lon", "lng"))
                    if not (city_key and lat_key and lon_key):
                        continue
                    try:
                        lat, lon = float(row[lat_key]), float(row[lon_key])
                    except (TypeError, ValueError):
                        continue
                    city = normal(row[city_key])
                    if city and 6 <= lat <= 38 and 68 <= lon <= 98:
                        cities.add(city.casefold())
        except (UnicodeDecodeError, csv.Error, OSError):
            continue
    return cities


def inspect(files: list[Path]) -> dict[str, Any]:
    image_files = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
    parquet = [p for p in files if p.suffix.lower() == ".parquet"]
    csvs = [p for p in files if p.suffix.lower() == ".csv"]
    jsons = [p for p in files if p.suffix.lower() == ".json"]
    schemas: dict[str, list[str]] = {}
    parquet_rows = 0
    try:
        import pyarrow.parquet as pq
        for path in parquet:
            pf = pq.ParquetFile(path)
            schemas[str(path.relative_to(REPO))] = pf.schema_arrow.names
            parquet_rows += pf.metadata.num_rows
    except ImportError:
        schemas = {str(p.relative_to(REPO)): ["pyarrow unavailable"] for p in parquet}
    return {"direct_images": len(image_files), "parquet_files": parquet,
            "parquet_rows": parquet_rows, "csv_files": csvs, "json_files": jsons,
            "parquet_schemas": schemas}


def csv_candidates(csv_files: list[Path]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for source in csv_files:
        try:
            with source.open("r", encoding="utf-8-sig", newline="") as handle:
                for index, row in enumerate(csv.DictReader(handle), start=2):
                    city_key, date_key, path_key = (find_key(row, CITY_KEYS), find_key(row, DATE_KEYS), find_key(row, PATH_KEYS))
                    if not (city_key and date_key and path_key):
                        continue
                    city, acquired = normal(row[city_key]), parse_date(row[date_key])
                    image = resolve_path(normal(row[path_key]), source)
                    if city and city.upper() != "UNKNOWN" and acquired and image:
                        location_key = find_key(row, LOCATION_KEYS)
                        found.append({"city": city, "date": acquired, "location_id": normal(row.get(location_key)) if location_key else "UNKNOWN",
                                      "source": str(source.relative_to(REPO)), "source_row": index, "path": image, "bytes": None})
        except (UnicodeDecodeError, csv.Error, OSError):
            continue
    return found


def parquet_candidates(paths: list[Path]) -> list[dict[str, Any]]:
    """Read only Parquet files that explicitly have city, date and image fields."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return []
    found: list[dict[str, Any]] = []
    for source in paths:
        pf = pq.ParquetFile(source)
        names = pf.schema_arrow.names
        lower = {n.lower(): n for n in names}
        city_key = next((lower[k] for k in CITY_KEYS if k in lower), None)
        date_key = next((lower[k] for k in DATE_KEYS if k in lower), None)
        image_key = lower.get("image") or lower.get("image_bytes")
        path_key = next((lower[k] for k in PATH_KEYS if k in lower), None)
        if not (city_key and date_key and (image_key or path_key)):
            continue
        columns = [city_key, date_key] + ([image_key] if image_key else [path_key])
        location_key = next((lower[k] for k in LOCATION_KEYS if k in lower), None)
        if location_key: columns.append(location_key)
        for batch in pf.iter_batches(columns=columns):
            for row in batch.to_pylist():
                city, acquired = normal(row[city_key]), parse_date(row[date_key])
                if not city or city.upper() == "UNKNOWN" or not acquired:
                    continue
                if image_key and row.get(image_key):
                    found.append({"city": city, "date": acquired, "location_id": normal(row.get(location_key)) if location_key else "UNKNOWN",
                                  "source": str(source.relative_to(REPO)), "source_row": None, "path": None, "bytes": row[image_key]})
                elif path_key:
                    image = resolve_path(normal(row[path_key]), source)
                    if image:
                        found.append({"city": city, "date": acquired, "location_id": normal(row.get(location_key)) if location_key else "UNKNOWN",
                                      "source": str(source.relative_to(REPO)), "source_row": None, "path": image, "bytes": None})
    return found


def select(candidates: list[dict[str, Any]], indian_cities: set[str]) -> list[dict[str, Any]]:
    by_city: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        if item["city"].casefold() in indian_cities:
            by_city[item["city"]].append(item)
    chosen: list[dict[str, Any]] = []
    # deterministic: cities with more valid dates first, then city name.
    ranked = sorted(by_city.items(), key=lambda pair: (-len({x["date"] for x in pair[1]}), pair[0].casefold()))[:20]
    for _, records in ranked:
        per_date = {d: next(x for x in records if x["date"] == d) for d in sorted({x["date"] for x in records})}
        dates = sorted(per_date)
        if len(dates) > 3:  # earliest, middle, latest: maximizes temporal spread simply and deterministically.
            dates = [dates[0], dates[len(dates) // 2], dates[-1]]
        chosen.extend(per_date[d] for d in dates)
    return chosen


def safe_city(city: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", city).strip("_") or "UNKNOWN"


def materialize(selected: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    counters: Counter[str] = Counter()
    image_root = OUTPUT / "images"
    for item in selected:
        city_display, city_slug = item["city"], safe_city(item["city"])
        counters[city_slug] += 1
        acquired = item["date"]
        if item["bytes"] is not None:
            try:
                with Image.open(io.BytesIO(item["bytes"])) as probe:
                    probe.verify()
                with Image.open(io.BytesIO(item["bytes"])) as probe:
                    suffix = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tiff", "WEBP": ".webp", "BMP": ".bmp"}.get(probe.format or "", ".img")
                if suffix == ".img":
                    continue
                destination = image_root / city_slug / f"{city_slug}_{counters[city_slug]:03d}_{acquired}{suffix}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(item["bytes"])
                with Image.open(destination) as probe:
                    probe.verify()
                path = destination
            except Exception:
                continue
        else:
            path = item["path"]
            try:
                with Image.open(path) as probe:
                    probe.verify()
            except Exception:
                continue
        rows.append({"image_id": f"{city_slug.upper()}_{counters[city_slug]:03d}_{acquired.replace('-', '')}", "city": city_display,
                     "location_id": item["location_id"] or "UNKNOWN", "image_path": path.relative_to(REPO).as_posix(),
                     "acquisition_date": acquired, "year": acquired[:4], "month": acquired[5:7]})
    return rows


def validate(rows: list[dict[str, str]]) -> tuple[int, int, list[str]]:
    errors: list[str] = []; opened = 0
    ids, paths = [r["image_id"] for r in rows], [r["image_path"] for r in rows]
    if len(ids) != len(set(ids)): errors.append("duplicate image IDs")
    if len(paths) != len(set(paths)): errors.append("duplicate image paths")
    for row in rows:
        try:
            with Image.open(REPO / row["image_path"]) as image: image.verify()
            opened += 1
        except Exception: errors.append(f"cannot open {row['image_path']}")
        if not parse_date(row["acquisition_date"]) or row["year"] != row["acquisition_date"][:4] or row["month"] != row["acquisition_date"][5:7]: errors.append(f"invalid date fields for {row['image_id']}")
    for city, group in __import__("itertools").groupby(sorted(rows, key=lambda r: r["city"]), key=lambda r: r["city"]):
        records = list(group)
        if len(records) > 3 or len({r["acquisition_date"] for r in records}) != len(records): errors.append(f"invalid temporal selection for {city}")
    return opened, len(rows) - opened, errors


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--inspect-only", action="store_true"); args = parser.parse_args()
    report = inspect(repository_files())
    indian_cities = indian_city_catalog(report["csv_files"])
    candidates = csv_candidates(report["csv_files"]) + parquet_candidates(report["parquet_files"])
    selected = select(candidates, indian_cities)
    if args.inspect_only:
        print(report); print(f"Valid city-and-date candidates: {len(candidates)}; selected: {len(selected)}"); return
    (OUTPUT / "metadata").mkdir(parents=True, exist_ok=True); (OUTPUT / "images").mkdir(parents=True, exist_ok=True)
    rows = materialize(selected); opened, failed, errors = validate(rows)
    csv_path = OUTPUT / "metadata" / "indian_cities_images.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "city", "location_id", "image_path", "acquisition_date", "year", "month"]); writer.writeheader(); writer.writerows(rows)
    by_city = Counter(row["city"] for row in rows)
    values = {"source_direct_images": report["direct_images"], "source_parquet_image_records": report["parquet_rows"], "source_parquet_files": len(report["parquet_files"]), "source_csv_files": len(report["csv_files"]), "source_json_files": len(report["json_files"]), "indian_city_labels_in_source_catalog": len(indian_cities), "cities_found_in_source_images": len({c["city"] for c in candidates if c["city"].casefold() in indian_cities}), "indian_cities_selected": len(by_city), "total_images_selected": len(rows), "images_with_real_dates": len(rows), "images_without_dates": 0, "cities_with_3_images": sum(n == 3 for n in by_city.values()), "cities_with_fewer_than_3_images": sum(n < 3 for n in by_city.values()), "images_successfully_opened": opened, "images_failed_validation": failed, "validation_errors": " | ".join(errors) or "none", "csv_location": csv_path.relative_to(REPO).as_posix()}
    with (OUTPUT / "metadata" / "dataset_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=values.keys()); writer.writeheader(); writer.writerow(values)
    print("=" * 30 + "\nINDIAN CITIES DATASET REPORT\n" + "=" * 30)
    print(f"Cities found in source dataset: {values['cities_found_in_source_images']}\nIndian cities selected: {values['indian_cities_selected']}\n\nTotal images selected: {values['total_images_selected']}\n\nImages with real dates: {values['images_with_real_dates']}\nImages without dates: {values['images_without_dates']}\n\nCities with 3 images: {values['cities_with_3_images']}\nCities with fewer than 3 images: {values['cities_with_fewer_than_3_images']}\n\nImages successfully opened: {opened}\nImages failed validation: {failed}\n\nCSV location:\n{values['csv_location']}")
    if errors: raise SystemExit("Validation failed: " + "; ".join(errors))


if __name__ == "__main__": main()
