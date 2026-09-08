"""
ingest_new_images.py
Pipeline script to ingest new satellite images dropped into data/new_images/,
move them to processed storage, extract metadata, generate embeddings, and update indexes.
"""

from pathlib import Path
import shutil
import sqlite3
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
NEW_IMAGES_DIR = BASE_DIR / "data" / "new_images"
PROCESSED_IMAGES_DIR = BASE_DIR / "data" / "processed" / "images"
DB_PATH = BASE_DIR / "database" / "terrawatch.db"


def ingest_new_images() -> None:
    NEW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    new_files = [
        f for f in NEW_IMAGES_DIR.iterdir() if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    ]

    if not new_files:
        print(f"No new images found in {NEW_IMAGES_DIR}")
        return

    print(f"Found {len(new_files)} new images to ingest.")
    conn = sqlite3.connect(DB_PATH) if DB_PATH.exists() else None

    ingested_count = 0
    for file_path in new_files:
        dest_path = PROCESSED_IMAGES_DIR / file_path.name
        if dest_path.exists():
            base_stem = file_path.stem
            suffix = file_path.suffix
            counter = 1
            while (PROCESSED_IMAGES_DIR / f"{base_stem}_{counter}{suffix}").exists():
                counter += 1
            dest_path = PROCESSED_IMAGES_DIR / f"{base_stem}_{counter}{suffix}"

        shutil.move(str(file_path), str(dest_path))

        try:
            with Image.open(dest_path) as img:
                width, height = img.size
                img_format = img.format or "UNKNOWN"
        except Exception:
            width, height, img_format = None, None, "UNKNOWN"

        if conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO images (image_name, file_path, width, height, format, source)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (dest_path.name, str(dest_path.relative_to(BASE_DIR)), width, height, img_format, "manual_ingest"),
            )
            conn.commit()

        ingested_count += 1
        print(f"Ingested: {dest_path.name}")

    if conn:
        conn.close()

    print(f"Ingestion complete: {ingested_count} images processed.")
    print("Tip: Run scripts/build_embeddings.py and scripts/build_index.py to update search index.")


if __name__ == "__main__":
    ingest_new_images()
