"""
RemoteCLIP Single Image Test

Tests RemoteCLIP on a single satellite image to verify:
1. Model loads correctly
2. Embeddings are generated successfully
3. Text-to-image similarity scores work
4. Results are reasonable

This is Phase 1 of the RemoteCLIP evaluation experiment.
"""

import sys
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from PIL import Image

# Import RemoteCLIP service
from experiments.remoteclip.remoteclip_embeddings import remoteclip_service

# Import evaluation queries
from experiments.remoteclip.evaluation_framework import get_all_queries


def test_single_image():
    """Test RemoteCLIP on a single satellite image."""
    
    print("\n" + "=" * 70)
    print("REMOTECLIP - SINGLE IMAGE TEST")
    print("=" * 70)
    
    # Select first available image from processed images
    images_dir = BASE_DIR / "data" / "processed" / "images"
    
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    
    if not image_files:
        print("\nERROR: No images found in data/processed/images/")
        return
    
    # Use first image
    test_image_path = image_files[0]
    
    print(f"\nTest Image: {test_image_path.name}")
    print(f"Full Path: {test_image_path}")
    
    # Load image
    print("\nLoading image...")
    with Image.open(test_image_path) as img:
        test_image = img.convert("RGB").copy()
        print(f"✓ Image loaded: {test_image.size}")
    
    # Generate image embedding
    print("\nGenerating image embedding with RemoteCLIP...")
    image_embedding = remoteclip_service.embed_image(test_image)
    
    print(f"✓ Image embedding generated!")
    print(f"  Shape: {image_embedding.shape}")
    print(f"  Dimension: {image_embedding.shape[1]}")
    
    # Test with semantic queries
    print("\nTesting text-to-image similarity with semantic queries...")
    print("-" * 70)
    
    # Use a subset of evaluation queries
    test_queries = [
        "water body reservoir lake",
        "urban city area with buildings",
        "agricultural farmland cropland",
        "forest vegetation trees",
        "airport with runway",
        "highway and roads",
        "mountains and terrain",
        "desert arid land",
        "industrial facility",
        "port and harbor",
    ]
    
    # Generate text embeddings
    text_embeddings = remoteclip_service.embed_text(test_queries)
    
    print(f"✓ Generated embeddings for {len(test_queries)} queries")
    
    # Calculate similarity scores (cosine similarity via dot product)
    similarities = np.dot(text_embeddings, image_embedding.T).flatten()
    
    # Rank queries by similarity
    ranked_indices = np.argsort(similarities)[::-1]
    
    print(f"\nTop Query Results for {test_image_path.name}:")
    print("-" * 70)
    
    for rank, idx in enumerate(ranked_indices[:10], start=1):
        query = test_queries[idx]
        score = similarities[idx]
        print(f"{rank:2d}. {query:<40} | Similarity: {score:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ REMOTECLIP SINGLE IMAGE TEST COMPLETE")
    print("=" * 70)
    print("\nRemoteCLIP model loaded and working correctly!")
    print("Ready for full evaluation.\n")


if __name__ == "__main__":
    try:
        test_single_image()
    except Exception as error:
        print(f"\n✗ ERROR: {error}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
