"""
retrieval.py

Semantic and similarity retrieval service for TerraWatch AI.

Supports:
1. Text -> satellite image search
2. Image -> similar satellite image search

Architecture:
    User Query (Text or Image)
        ↓
    EmbeddingService (OpenCLIP ViT-B-32)
        ↓
    Normalized 512-dimensional vector
        ↓
    FAISS IndexFlatIP (Cosine Similarity)
        ↓
    Ranked Satellite Imagery Results
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
from typing import Any
import numpy as np
from PIL import Image

from backend.config import settings
from backend.embeddings import embedding_service


class VectorRetriever:
    """
    Handles semantic retrieval using the TerraWatch FAISS index.
    """

    def __init__(self) -> None:
        self.index = None
        self.image_ids: list[str] = []
        self.image_paths: list[str] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # LOAD INDEX
    # ------------------------------------------------------------------

    def load_index(self) -> bool:
        """
        Load the FAISS index and image mappings.

        Returns:
            True  -> index loaded successfully
            False -> loading failed
        """
        if self._loaded and self.index is not None:
            return True

        # Find FAISS index file (support both satellite.faiss and satellite_faiss.index)
        faiss_path = settings.FAISS_INDEX_PATH
        if not faiss_path.exists():
            alternate_names = [
                settings.INDEXES_DIR / "satellite.faiss",
                settings.INDEXES_DIR / "satellite_faiss.index",
            ]
            for alt in alternate_names:
                if alt.exists():
                    faiss_path = alt
                    break

        image_ids_path = getattr(settings, "IMAGE_IDS_PATH", settings.INDEXES_DIR / "image_ids.json")
        image_paths_path = getattr(
            settings,
            "IMAGE_PATHS_PATH",
            getattr(settings, "PATHS_MAP_PATH", settings.INDEXES_DIR / "image_paths.json"),
        )

        # --------------------------------------------------------------
        # Check required files
        # --------------------------------------------------------------
        if not faiss_path.exists():
            print("\nERROR: FAISS index not found.")
            print(f"Expected: {faiss_path}")
            print("\nRun: python scripts/build_index.py")
            return False

        if not image_ids_path.exists():
            print("\nERROR: Image ID mapping not found.")
            print(f"Expected: {image_ids_path}")
            print("\nRun: python scripts/build_embeddings.py")
            return False

        # --------------------------------------------------------------
        # Load FAISS
        # --------------------------------------------------------------
        try:
            import faiss

            print(f"Loading FAISS index from {faiss_path.name}...")
            self.index = faiss.read_index(str(faiss_path))
        except ImportError:
            print("\nERROR: FAISS is not installed. Install with: pip install faiss-cpu")
            return False
        except Exception as error:
            print(f"\nERROR: Failed to load FAISS index: {error}")
            return False

        # --------------------------------------------------------------
        # Load image IDs
        # --------------------------------------------------------------
        try:
            with open(image_ids_path, "r", encoding="utf-8") as file:
                self.image_ids = json.load(file)
        except Exception as error:
            print(f"\nERROR: Failed to load image IDs: {error}")
            return False

        # --------------------------------------------------------------
        # Load image paths if available
        # --------------------------------------------------------------
        if image_paths_path is not None and Path(image_paths_path).exists():
            try:
                with open(image_paths_path, "r", encoding="utf-8") as file:
                    self.image_paths = json.load(file)
            except Exception as error:
                print(f"Warning: Could not load image paths: {error}")

        # --------------------------------------------------------------
        # Validate mappings
        # --------------------------------------------------------------
        if self.index is None:
            print("ERROR: FAISS index is None.")
            return False

        if self.index.ntotal != len(self.image_ids):
            print("\nERROR: FAISS index and image ID mapping count mismatch.")
            print(f"FAISS vectors: {self.index.ntotal}")
            print(f"Image IDs:     {len(self.image_ids)}")
            return False

        self._loaded = True
        print(f"FAISS index loaded successfully ({self.index.ntotal} vectors, dimension {self.index.d})")
        return True

    # ------------------------------------------------------------------
    # VALIDATE SEARCH PARAMETERS
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        if not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        return top_k

    # ------------------------------------------------------------------
    # FORMAT RESULTS
    # ------------------------------------------------------------------

    def _format_results(
        self,
        scores: np.ndarray,
        indices: np.ndarray,
    ) -> list[dict[str, Any]]:
        """
        Convert FAISS results into clean Python dictionaries.
        """
        results: list[dict[str, Any]] = []

        for score, index in zip(scores[0], indices[0]):
            idx = int(index)
            if idx < 0 or idx >= len(self.image_ids):
                continue

            result: dict[str, Any] = {
                "image_id": self.image_ids[idx],
                "score": float(score),
                "index": idx,
            }

            # Add image path when available
            if idx < len(self.image_paths):
                result["path"] = self.image_paths[idx]
            
            # Add image URL for frontend display
            result["image_url"] = f"/api/v1/images/{self.image_ids[idx]}/file"

            results.append(result)

        return results

    # ------------------------------------------------------------------
    # SEARCH BY TEXT
    # ------------------------------------------------------------------

    def search_by_text(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search satellite images using a natural-language query.

        Example:
            search_by_text("deep water lake reservoir", top_k=10)
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty.")

        if top_k is None:
            top_k = settings.DEFAULT_TOP_K

        top_k = self._validate_top_k(top_k)

        if not self.load_index():
            return []

        # Generate normalized text embedding
        query_vector = embedding_service.embed_text(query)

        if query_vector.ndim != 2:
            query_vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)

        query_vector = query_vector.astype(np.float32, copy=False)

        if not np.isfinite(query_vector).all():
            raise ValueError("Text embedding contains NaN or infinite values.")

        import faiss
        faiss.normalize_L2(query_vector)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)

        return self._format_results(scores, indices)

    # ------------------------------------------------------------------
    # SEARCH BY IMAGE
    # ------------------------------------------------------------------

    def search_by_image(
        self,
        image: Image.Image | str | Path,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for satellite images similar to an input image.

        Accepts:
            PIL.Image.Image, file path str, or pathlib.Path
        """
        if top_k is None:
            top_k = settings.DEFAULT_TOP_K

        top_k = self._validate_top_k(top_k)

        if not self.load_index():
            return []

        # Load image when a path is supplied
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            with Image.open(image_path) as img:
                pil_image = img.convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise TypeError("image must be a PIL Image, string path, or pathlib.Path.")

        # Generate normalized image embedding
        query_vector = embedding_service.embed_image(pil_image)

        if query_vector.ndim != 2:
            query_vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)

        query_vector = query_vector.astype(np.float32, copy=False)

        if not np.isfinite(query_vector).all():
            raise ValueError("Image embedding contains NaN or infinite values.")

        import faiss
        faiss.normalize_L2(query_vector)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)

        return self._format_results(scores, indices)


# ----------------------------------------------------------------------
# GLOBAL RETRIEVER INSTANCE
# ----------------------------------------------------------------------

retriever = VectorRetriever()


# ----------------------------------------------------------------------
# SELF TEST / CLI DEMO
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TERRAWATCH AI - SEMANTIC RETRIEVAL TEST")
    print("=" * 70)

    # 1. Text Search Test
    sample_query = "urban city area with buildings and road network"
    print(f"\n[Test 1] Running text search for query: '{sample_query}'")

    results = retriever.search_by_text(sample_query, top_k=5)
    print(f"Retrieved {len(results)} results:\n")

    for rank, res in enumerate(results, start=1):
        image_id = res.get("image_id", "N/A")
        score = res.get("score", 0.0)
        path = res.get("path", "N/A")
        print(f"  {rank}. ID: {image_id:<12} Score: {score:.4f}  Path: {path}")

    # 2. Image-to-Image Search Test (if image paths available)
    if results and "path" in results[0]:
        query_img_path = ROOT_DIR / results[0]["path"]
        if query_img_path.exists():
            print(f"\n[Test 2] Running image-to-image similarity search using: {query_img_path.name}")
            img_results = retriever.search_by_image(query_img_path, top_k=3)
            print(f"Retrieved {len(img_results)} similar images:\n")

            for rank, res in enumerate(img_results, start=1):
                image_id = res.get("image_id", "N/A")
                score = res.get("score", 0.0)
                path = res.get("path", "N/A")
                print(f"  {rank}. ID: {image_id:<12} Score: {score:.4f}  Path: {path}")

    print("\n" + "=" * 70)
    print("RETRIEVAL TEST COMPLETE")
    print("=" * 70 + "\n")
