"""
similar_locations.py

Similar Location Discovery Service

Finds geographically different satellite locations with similar visual/semantic characteristics.
Uses existing FAISS index, embeddings, and metadata.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import math

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np

from backend.config import settings
from backend.retrieval import retriever


# ============================================================
# HAVERSINE DISTANCE CALCULATION
# ============================================================

def calculate_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calculate the great-circle distance between two points on Earth using Haversine formula.
    
    Args:
        lat1: Latitude of point 1 (degrees)
        lon1: Longitude of point 1 (degrees)
        lat2: Latitude of point 2 (degrees)
        lon2: Longitude of point 2 (degrees)
    
    Returns:
        Distance in kilometers
    """
    # Earth radius in kilometers
    R = 6371.0
    
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance


# ============================================================
# SIMILAR LOCATION DISCOVERY SERVICE
# ============================================================

class SimilarLocationService:
    """
    Service for finding geographically different locations with similar visual characteristics.
    """
    
    def __init__(self):
        self.metadata_df: Optional[pd.DataFrame] = None
        self._metadata_loaded = False
    
    def _load_metadata(self) -> None:
        """Load metadata CSV with image locations."""
        if self._metadata_loaded and self.metadata_df is not None:
            return
        
        metadata_path = settings.PROCESSED_METADATA_DIR / "metadata.csv"
        
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}\n"
                f"Run scripts/extract_images.py to generate metadata."
            )
        
        try:
            self.metadata_df = pd.read_csv(metadata_path)
            
            # Validate required columns
            required_cols = ["image_id", "latitude", "longitude", "image_path"]
            missing_cols = [col for col in required_cols if col not in self.metadata_df.columns]
            
            if missing_cols:
                raise ValueError(f"Metadata missing required columns: {missing_cols}")
            
            # Create image_id index for fast lookup
            self.metadata_df.set_index("image_id", inplace=True)
            
            self._metadata_loaded = True
            print(f"Metadata loaded: {len(self.metadata_df)} images")
            
        except Exception as error:
            raise RuntimeError(f"Failed to load metadata: {error}") from error
    
    def get_image_metadata(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific image.
        
        Args:
            image_id: Image identifier (e.g., "S2_000001")
        
        Returns:
            Dictionary with image metadata or None if not found
        """
        self._load_metadata()
        
        if image_id not in self.metadata_df.index:
            return None
        
        row = self.metadata_df.loc[image_id]
        
        return {
            "image_id": image_id,
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "image_path": str(row["image_path"]),
        }
    
    def find_similar_locations(
        self,
        source_image_id: str,
        top_k: int = 10,
        min_distance_km: float = 0.5,
        search_candidates: int = 50,
    ) -> Dict[str, Any]:
        """
        Find geographically different locations with similar visual/semantic characteristics.
        
        Args:
            source_image_id: Source image ID (e.g., "S2_000001")
            top_k: Number of similar locations to return (1-50)
            min_distance_km: Minimum geographic distance to filter near-duplicates (km)
            search_candidates: Number of FAISS candidates to retrieve before filtering
        
        Returns:
            Dictionary with source location info and similar locations
        """
        # Validate parameters
        if not isinstance(top_k, int) or top_k < 1 or top_k > 50:
            raise ValueError("top_k must be between 1 and 50")
        
        if min_distance_km < 0:
            raise ValueError("min_distance_km must be >= 0")
        
        # Load metadata
        self._load_metadata()
        
        # Get source image metadata
        source_metadata = self.get_image_metadata(source_image_id)
        
        if source_metadata is None:
            raise ValueError(f"Image ID not found: {source_image_id}")
        
        source_lat = source_metadata["latitude"]
        source_lon = source_metadata["longitude"]
        
        # Ensure retriever is loaded
        if not retriever.load_index():
            raise RuntimeError("Failed to load FAISS index")
        
        # Find the source image in the FAISS index
        # The image_ids list maps FAISS vector index → image_id
        try:
            source_faiss_idx = retriever.image_ids.index(source_image_id)
        except ValueError:
            raise ValueError(f"Image ID not found in FAISS index: {source_image_id}")
        
        # Get the embedding for the source image from FAISS
        # Use FAISS reconstruct to get the original vector
        import faiss
        source_embedding = retriever.index.reconstruct(source_faiss_idx)
        source_embedding = source_embedding.reshape(1, -1).astype(np.float32)
        
        # Ensure normalized (should already be, but double-check)
        faiss.normalize_L2(source_embedding)
        
        # Search FAISS for candidates (get more than needed for filtering)
        k = min(search_candidates, retriever.index.ntotal)
        scores, indices = retriever.index.search(source_embedding, k)
        
        # Process results
        similar_locations = []
        
        for score, idx in zip(scores[0], indices[0]):
            idx = int(idx)
            
            # Skip invalid indices
            if idx < 0 or idx >= len(retriever.image_ids):
                continue
            
            candidate_id = retriever.image_ids[idx]
            
            # Skip the source image itself
            if candidate_id == source_image_id:
                continue
            
            # Get candidate metadata
            candidate_metadata = self.get_image_metadata(candidate_id)
            
            if candidate_metadata is None:
                continue
            
            candidate_lat = candidate_metadata["latitude"]
            candidate_lon = candidate_metadata["longitude"]
            
            # Skip if missing coordinates
            if pd.isna(candidate_lat) or pd.isna(candidate_lon):
                continue
            
            # Calculate geographic distance
            distance_km = calculate_distance_km(
                source_lat, source_lon,
                candidate_lat, candidate_lon
            )
            
            # Filter by minimum distance (remove near-duplicates)
            if distance_km < min_distance_km:
                continue
            
            # Add to results
            similar_locations.append({
                "image_id": candidate_id,
                "latitude": candidate_lat,
                "longitude": candidate_lon,
                "image_path": candidate_metadata["image_path"],
                "image_url": f"/api/v1/images/{candidate_id}/file",
                "semantic_similarity": float(score),  # Cosine similarity (0-1)
                "distance_km": round(distance_km, 2),
            })
        
        # Sort by semantic similarity (descending)
        similar_locations.sort(key=lambda x: x["semantic_similarity"], reverse=True)
        
        # Take top K
        similar_locations = similar_locations[:top_k]
        
        # Add rank
        for rank, location in enumerate(similar_locations, start=1):
            location["rank"] = rank
        
        # Build response
        response = {
            "success": True,
            "source_location": {
                "image_id": source_image_id,
                "latitude": source_lat,
                "longitude": source_lon,
                "image_path": source_metadata["image_path"],
                "image_url": f"/api/v1/images/{source_image_id}/file",
            },
            "parameters": {
                "top_k": top_k,
                "min_distance_km": min_distance_km,
                "search_candidates": search_candidates,
            },
            "total_results": len(similar_locations),
            "similar_locations": similar_locations,
        }
        
        return response


# ============================================================
# GLOBAL SERVICE INSTANCE
# ============================================================

similar_location_service = SimilarLocationService()


# ============================================================
# TEST / DEMO
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SIMILAR LOCATION DISCOVERY - TEST")
    print("=" * 70)
    
    # Test with first image
    test_image_id = "S2_000001"
    
    print(f"\nFinding similar locations for: {test_image_id}")
    
    try:
        results = similar_location_service.find_similar_locations(
            source_image_id=test_image_id,
            top_k=5,
            min_distance_km=1.0,
        )
        
        print(f"\nSource Location:")
        print(f"  ID: {results['source_location']['image_id']}")
        print(f"  Lat: {results['source_location']['latitude']}")
        print(f"  Lon: {results['source_location']['longitude']}")
        
        print(f"\nFound {results['total_results']} similar locations:\n")
        
        for loc in results["similar_locations"]:
            print(f"  {loc['rank']}. {loc['image_id']}")
            print(f"     Similarity: {loc['semantic_similarity']:.4f}")
            print(f"     Distance: {loc['distance_km']} km")
            print(f"     Location: ({loc['latitude']}, {loc['longitude']})")
            print()
        
        print("=" * 70)
        print("TEST COMPLETE")
        print("=" * 70 + "\n")
        
    except Exception as error:
        print(f"\nERROR: {error}")
        import traceback
        traceback.print_exc()
