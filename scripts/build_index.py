
"""
build_index.py

Builds a FAISS vector index from precomputed satellite-image embeddings.

Input:
    indexes/embeddings.npy

Output:
    indexes/satellite.faiss
    indexes/faiss_metadata.json

The embeddings are normalized and indexed using Inner Product,
which is equivalent to cosine similarity for normalized vectors.
"""

from pathlib import Path
import json

import numpy as np


# ---------------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_DIR = BASE_DIR / "indexes"

EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
FAISS_INDEX_PATH = INDEX_DIR / "satellite.faiss"
FAISS_METADATA_PATH = INDEX_DIR / "faiss_metadata.json"


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# BUILD FAISS INDEX
# ---------------------------------------------------------------------------

def build_faiss_index(
    embeddings_path: Path = EMBEDDINGS_PATH,
    index_path: Path = FAISS_INDEX_PATH,
    metadata_path: Path = FAISS_METADATA_PATH,
) -> None:

    print()
    print("=" * 70)
    print("TERRAWATCH AI - FAISS INDEX BUILDING")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. CHECK EMBEDDINGS FILE
    # -----------------------------------------------------------------------

    if not embeddings_path.exists():
        print()
        print("ERROR: Embeddings file not found.")
        print(f"Expected location: {embeddings_path}")
        print()
        print("Run this first:")
        print("python scripts/build_embeddings.py")
        return

    # -----------------------------------------------------------------------
    # 2. LOAD EMBEDDINGS
    # -----------------------------------------------------------------------

    print()
    print("Loading embeddings...")
    print(f"File: {embeddings_path}")

    try:
        embeddings = np.load(embeddings_path)
    except Exception as error:
        print()
        print(f"ERROR: Could not load embeddings: {error}")
        return

    # -----------------------------------------------------------------------
    # 3. VALIDATE EMBEDDING SHAPE
    # -----------------------------------------------------------------------

    if embeddings.ndim != 2:
        print()
        print(
            f"ERROR: Expected a 2D embedding matrix, "
            f"but received shape {embeddings.shape}"
        )
        return

    num_vectors, dimension = embeddings.shape

    if num_vectors == 0:
        print()
        print("ERROR: Embedding file contains zero vectors.")
        return

    if dimension == 0:
        print()
        print("ERROR: Embedding dimension is zero.")
        return

    print()
    print(f"Number of vectors: {num_vectors}")
    print(f"Embedding dimension: {dimension}")

    # -----------------------------------------------------------------------
    # 4. CONVERT TO FLOAT32
    # -----------------------------------------------------------------------

    embeddings = embeddings.astype(np.float32, copy=False)

    print(f"Data type: {embeddings.dtype}")

    # -----------------------------------------------------------------------
    # 5. CHECK FOR INVALID VALUES
    # -----------------------------------------------------------------------

    if not np.isfinite(embeddings).all():
        print()
        print("ERROR: Embeddings contain NaN or infinite values.")
        print("The FAISS index cannot be built safely.")
        return

    # -----------------------------------------------------------------------
    # 6. LOAD FAISS
    # -----------------------------------------------------------------------

    try:
        import faiss
    except ImportError:
        print()
        print("ERROR: FAISS is not installed.")
        print()
        print("Install it with:")
        print("pip install faiss-cpu")
        return

    # -----------------------------------------------------------------------
    # 7. NORMALIZE EMBEDDINGS
    # -----------------------------------------------------------------------
    #
    # For normalized vectors:
    #
    #       Inner Product = Cosine Similarity
    #
    # This makes IndexFlatIP suitable for semantic similarity search.
    # -----------------------------------------------------------------------

    print()
    print("Normalizing embeddings for cosine similarity...")

    faiss.normalize_L2(embeddings)

    # -----------------------------------------------------------------------
    # 8. CREATE FAISS INDEX
    # -----------------------------------------------------------------------

    print("Creating FAISS IndexFlatIP index...")

    index = faiss.IndexFlatIP(dimension)

    # -----------------------------------------------------------------------
    # 9. ADD VECTORS
    # -----------------------------------------------------------------------

    print("Adding vectors to FAISS index...")

    index.add(embeddings)

    # -----------------------------------------------------------------------
    # 10. VERIFY INDEX
    # -----------------------------------------------------------------------

    if index.ntotal != num_vectors:
        print()
        print("ERROR: FAISS vector count does not match embeddings count.")
        print(f"Expected: {num_vectors}")
        print(f"Indexed:  {index.ntotal}")
        return

    # -----------------------------------------------------------------------
    # 11. SAVE FAISS INDEX
    # -----------------------------------------------------------------------

    faiss.write_index(
        index,
        str(index_path),
    )

    # -----------------------------------------------------------------------
    # 12. SAVE INDEX METADATA
    # -----------------------------------------------------------------------

    metadata = {
        "index_type": "IndexFlatIP",
        "similarity": "cosine",
        "num_vectors": int(num_vectors),
        "dimension": int(dimension),
        "embedding_file": str(
            embeddings_path.relative_to(BASE_DIR)
        ),
        "faiss_index_file": str(
            index_path.relative_to(BASE_DIR)
        ),
        "normalized": True,
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    # -----------------------------------------------------------------------
    # 13. FINAL REPORT
    # -----------------------------------------------------------------------

    print()
    print("=" * 70)
    print("FAISS INDEX CREATED SUCCESSFULLY")
    print("=" * 70)

    print()
    print(f"Total vectors:       {index.ntotal}")
    print(f"Vector dimension:    {dimension}")
    print("Similarity:          Cosine similarity")
    print("Index type:          IndexFlatIP")

    print()
    print("Saved FAISS index:")
    print(index_path)

    print()
    print("Saved metadata:")
    print(metadata_path)

    print()
    print("=" * 70)
    print("INDEX BUILD COMPLETE")
    print("=" * 70)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_faiss_index()

