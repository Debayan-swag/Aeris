"""
TerraWatch AI - Build Image Embeddings

Generates normalized embeddings for all processed satellite images using
the centralized EmbeddingService.

Outputs:
- indexes/embeddings.npy
- indexes/image_ids.json
- indexes/image_paths.json
- data/processed/metadata/embedding_errors.json (if any failures occur)
"""

from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ============================================================
# TERRAWATCH IMPORTS
# ============================================================

from backend.config import settings
from backend.embeddings import embedding_service

# ============================================================
# SUPPORTED IMAGE FORMATS
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}

# ============================================================
# CONFIGURATION
# ============================================================

IMAGES_DIR = settings.PROCESSED_IMAGES_DIR
OUTPUT_DIR = settings.INDEXES_DIR

EMBEDDINGS_PATH = settings.EMBEDDINGS_PATH
IMAGE_IDS_PATH = settings.IMAGE_IDS_PATH
PATHS_MAP_PATH = settings.PATHS_MAP_PATH

BATCH_SIZE = 32


# ============================================================
# FIND IMAGES
# ============================================================

def get_image_paths(images_dir: Path) -> list[Path]:
    """
    Return sorted list of valid satellite image paths from images_dir.
    """
    image_paths = [
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(image_paths)


# ============================================================
# BUILD EMBEDDINGS
# ============================================================

def build_embeddings(
    images_dir: Path = IMAGES_DIR,
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    Generate normalized embeddings for all satellite images in images_dir.
    """
    print("\n" + "=" * 70)
    print("TERRAWATCH AI - IMAGE EMBEDDING GENERATION")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK IMAGE DIRECTORY
    # --------------------------------------------------------
    if not images_dir.exists():
        print(
            f"\nERROR: Image directory does not exist:\n{images_dir}\n"
            "\nRun extract_images.py first."
        )
        return

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings.PROCESSED_METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # FIND IMAGES
    # --------------------------------------------------------
    image_paths = get_image_paths(images_dir)

    if not image_paths:
        print(f"\nNo supported images found in:\n{images_dir}")
        return

    print(f"\nImages found: {len(image_paths)}")
    print(f"Batch size: {batch_size}")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------
    print("\nInitializing embedding service...")
    embedding_service._lazy_load()

    print(f"Model: {embedding_service.model_name}")
    print(f"Pretrained: {embedding_service.pretrained}")
    print(f"Device: {embedding_service._device}")

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------
    all_embeddings = []
    image_ids = []
    image_relative_paths = []
    failed_images = []

    # --------------------------------------------------------
    # PROCESS BATCHES
    # --------------------------------------------------------
    for start in tqdm(
        range(0, len(image_paths), batch_size),
        desc="Generating embeddings",
        unit="batch",
    ):
        batch_paths = image_paths[start : start + batch_size]

        batch_images = []
        batch_ids = []
        batch_rel_paths = []
        batch_paths_valid = []

        # ----------------------------------------------------
        # LOAD BATCH
        # ----------------------------------------------------
        for image_path in batch_paths:
            try:
                with Image.open(image_path) as image:
                    image_copy = image.convert("RGB").copy()

                batch_images.append(image_copy)
                batch_ids.append(image_path.stem)
                batch_rel_paths.append(str(image_path.relative_to(BASE_DIR)).replace("\\", "/"))
                batch_paths_valid.append(image_path)

            except Exception as error:
                failed_images.append(
                    {
                        "image_id": image_path.stem,
                        "image_path": str(image_path),
                        "error": str(error),
                    }
                )
                print(f"\nFailed to load {image_path.name}: {error}")

        if not batch_images:
            continue

        # ----------------------------------------------------
        # GENERATE BATCH EMBEDDINGS
        # ----------------------------------------------------
        try:
            embeddings = embedding_service.embed_image(batch_images)

            if len(embeddings) != len(batch_ids):
                raise RuntimeError("Embedding count does not match image count.")

            all_embeddings.append(embeddings)
            image_ids.extend(batch_ids)
            image_relative_paths.extend(batch_rel_paths)

        except Exception as error:
            print(
                f"\nFailed to generate embeddings for batch starting at image {start + 1}: {error}"
            )
            for image_path in batch_paths_valid:
                failed_images.append(
                    {
                        "image_id": image_path.stem,
                        "image_path": str(image_path),
                        "error": str(error),
                    }
                )

    # --------------------------------------------------------
    # CHECK RESULTS
    # --------------------------------------------------------
    if not all_embeddings:
        print("\nERROR: No embeddings were generated.")
        return

    # --------------------------------------------------------
    # COMBINE EMBEDDINGS
    # --------------------------------------------------------
    embeddings_matrix = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------
    if len(embeddings_matrix) != len(image_ids):
        raise RuntimeError("Number of embeddings does not match number of image IDs.")

    if not np.isfinite(embeddings_matrix).all():
        raise RuntimeError("Embeddings contain NaN or infinite values.")

    # --------------------------------------------------------
    # SAVE EMBEDDINGS & INDEX MAPPINGS
    # --------------------------------------------------------
    np.save(EMBEDDINGS_PATH, embeddings_matrix)

    with open(IMAGE_IDS_PATH, "w", encoding="utf-8") as file:
        json.dump(image_ids, file, indent=2)

    with open(PATHS_MAP_PATH, "w", encoding="utf-8") as file:
        json.dump(image_relative_paths, file, indent=2)

    # --------------------------------------------------------
    # SAVE FAILED IMAGE REPORT
    # --------------------------------------------------------
    failed_report_path = settings.PROCESSED_METADATA_DIR / "embedding_errors.json"
    with open(failed_report_path, "w", encoding="utf-8") as file:
        json.dump(failed_images, file, indent=2)

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 70)

    print(f"\nTotal images processed: {len(image_paths)}")
    print(f"Successful embeddings: {len(image_ids)}")
    print(f"Failed images: {len(failed_images)}")
    print(f"\nEmbedding matrix shape: {embeddings_matrix.shape}")
    print(f"Embedding dimension: {embeddings_matrix.shape[1]}")
    print(f"Data type: {embeddings_matrix.dtype}")
    print(f"\nSaved embeddings:\n{EMBEDDINGS_PATH}")
    print(f"\nSaved image ID mapping:\n{IMAGE_IDS_PATH}")
    print(f"\nSaved image path mapping:\n{PATHS_MAP_PATH}")
    print(f"\nSaved error report:\n{failed_report_path}")
    print("\n" + "=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    build_embeddings()
