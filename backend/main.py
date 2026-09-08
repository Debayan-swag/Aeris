"""
main.py

TerraWatch AI FastAPI Backend

Currently available features:

1. Health check
2. Text -> Satellite Image Semantic Search
3. Image -> Similar Satellite Image Search
"""

from io import BytesIO
from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from backend.config import settings
from backend.retrieval import retriever
from backend.similar_locations import similar_location_service


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "TerraWatch AI - Semantic Retrieval and "
        "Satellite Image Similarity Search"
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# UPLOAD DIRECTORY
# ============================================================

UPLOAD_DIR = settings.DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": "1.0.0",
        "docs": "/docs",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
    }


# ============================================================
# TEXT SEMANTIC SEARCH
# ============================================================

@app.get("/api/v1/search/text")
def search_by_text(
    query: str = Query(
        ...,
        min_length=1,
        description="Natural language search query",
        examples=["urban area with buildings and roads"],
    ),
    top_k: int = Query(
        default=settings.DEFAULT_TOP_K,
        ge=1,
        le=50,
        description="Number of results to return",
    ),
):
    """
    Search satellite images using natural language.

    Example:

        /api/v1/search/text?query=urban%20area%20with%20buildings&top_k=5
    """
    try:
        results = retriever.search_by_text(
            query=query,
            top_k=top_k,
        )

        return {
            "query": query,
            "count": len(results),
            "results": results,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Semantic search failed: {str(error)}",
        )


# ============================================================
# IMAGE SIMILARITY SEARCH
# ============================================================

@app.post("/api/v1/search/image")
async def search_by_image(
    file: UploadFile = File(
        ...,
        description="Satellite image to search for similar images",
    ),
    top_k: int = Query(
        default=settings.DEFAULT_TOP_K,
        ge=1,
        le=50,
        description="Number of similar images to return",
    ),
):
    """
    Upload a satellite image and find similar images.

    Pipeline:
        Uploaded Image
            â†“
        Saved with unique UUID to data/uploads/
            â†“
        Validated (format, file size, PIL integrity check)
            â†“
        OpenCLIP Embedding (via retriever.search_by_image)
            â†“
        FAISS Cosine Similarity Search
            â†“
        Similar Satellite Images
    """
    # 1. Validate file extension and content type
    original_filename = file.filename or "upload.jpg"
    extension = Path(original_filename).suffix.lower()

    if not extension:
        extension = ".jpg"

    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported image file extension '{extension}'. "
                f"Supported formats: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"
            ),
        )

    if file.content_type and not (
        file.content_type.startswith("image/")
        or file.content_type == "application/octet-stream"
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be a valid image.",
        )

    # 2. Save with unique UUID
    file_id = uuid.uuid4().hex
    saved_filename = f"{file_id}{extension}"
    file_path = UPLOAD_DIR / saved_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. Check for empty file
        if file_path.stat().st_size == 0:
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        # 4. Verify image file integrity
        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception as img_err:
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail=f"Corrupted or invalid image file: {str(img_err)}",
            )

        # 5. Search using retriever
        results = retriever.search_by_image(
            image=file_path,
            top_k=top_k,
        )

        return {
            "filename": original_filename,
            "saved_path": str(file_path.relative_to(settings.BASE_DIR)).replace("\\", "/"),
            "count": len(results),
            "results": results,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Image similarity search failed: {str(error)}",
        )




# ============================================================
# SIMILAR LOCATION DISCOVERY
# ============================================================

@app.get("/api/v1/search/similar-locations/{image_id}")
def search_similar_locations(
    image_id: str,
    top_k: int = Query(
        default=10,
        ge=1,
        le=50,
        description="Number of similar locations to return",
    ),
    min_distance_km: float = Query(
        default=0.5,
        ge=0.0,
        description="Minimum geographic distance in km to filter near-duplicates",
    ),
):
    """
    Find geographically different locations with similar visual/semantic characteristics.
    
    This endpoint:
    1. Takes a source image ID
    2. Retrieves its embedding from the existing FAISS index
    3. Finds visually similar satellite images
    4. Filters by geographic distance to remove near-duplicates
    5. Returns semantically similar but geographically different locations
    
    Example:
        /api/v1/search/similar-locations/S2_000001?top_k=10&min_distance_km=1.0
    
    Use cases:
    - Find similar terrain types in different regions
    - Discover analogous land use patterns
    - Identify comparable infrastructure in other areas
    """
    try:
        results = similar_location_service.find_similar_locations(
            source_image_id=image_id,
            top_k=top_k,
            min_distance_km=min_distance_km,
        )
        
        return results
    
    except ValueError as error:
        # Invalid image ID or parameter validation error
        raise HTTPException(
            status_code=404 if "not found" in str(error).lower() else 400,
            detail=str(error),
        )
    
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Similar location search failed: {str(error)}",
        )



# ============================================================
# UPLOAD IMAGE AND FIND SIMILAR LOCATIONS
# ============================================================

@app.post("/api/v1/search/similar-locations-by-image")
async def search_similar_locations_by_image(
    file: UploadFile = File(
        ...,
        description="Upload any satellite/aerial image to find similar locations",
    ),
    top_k: int = Query(
        default=10,
        ge=1,
        le=50,
        description="Number of similar locations to return",
    ),
    min_distance_km: float = Query(
        default=0.5,
        ge=0.0,
        description="Minimum geographic distance in km to filter near-duplicates",
    ),
):
    """
    Upload ANY satellite/aerial image and find similar locations in the dataset.
    
    This endpoint:
    1. Accepts any uploaded image (jpg, png, tif)
    2. Generates embedding using OpenCLIP
    3. Searches FAISS for visually similar images
    4. Returns results with geographic coordinates
    5. Filters by minimum distance
    
    Example usage:
        curl -X POST "http://localhost:8000/api/v1/search/similar-locations-by-image?top_k=5" \
             -F "file=@/path/to/your/image.jpg"
    
    Use cases:
    - Upload a Google Earth screenshot and find similar terrains
    - Upload drone imagery and find comparable locations
    - Upload any satellite image and discover analogous sites
    """
    # Validate file extension
    original_filename = file.filename or "upload.jpg"
    extension = Path(original_filename).suffix.lower()
    
    if not extension:
        extension = ".jpg"
    
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{extension}'. Supported: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"
        )
    
    # Save uploaded file temporarily
    file_id = uuid.uuid4().hex
    saved_filename = f"{file_id}{extension}"
    file_path = UPLOAD_DIR / saved_filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Verify image integrity
        if file_path.stat().st_size == 0:
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Uploaded image is empty")
        
        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception as img_err:
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Corrupted or invalid image: {str(img_err)}")
        
        # Load metadata
        similar_location_service._load_metadata()
        
        # Ensure retriever is loaded
        if not retriever.load_index():
            raise RuntimeError("Failed to load FAISS index")
        
        # Load and process uploaded image
        with Image.open(file_path) as uploaded_img:
            uploaded_img_rgb = uploaded_img.convert("RGB").copy()
        
        # Generate embedding for uploaded image
        from backend.embeddings import embedding_service
        query_embedding = embedding_service.embed_image(uploaded_img_rgb)
        
        # Ensure proper format
        import numpy as np
        if query_embedding.ndim != 2:
            query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32, copy=False)
        
        # Normalize
        import faiss
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS
        search_candidates = min(top_k * 5, retriever.index.ntotal)
        scores, indices = retriever.index.search(query_embedding, search_candidates)
        
        # Process results with location filtering
        similar_locations = []
        
        for score, idx in zip(scores[0], indices[0]):
            idx = int(idx)
            
            if idx < 0 or idx >= len(retriever.image_ids):
                continue
            
            candidate_id = retriever.image_ids[idx]
            candidate_metadata = similar_location_service.get_image_metadata(candidate_id)
            
            if candidate_metadata is None:
                continue
            
            candidate_lat = candidate_metadata["latitude"]
            candidate_lon = candidate_metadata["longitude"]
            
            if pd.isna(candidate_lat) or pd.isna(candidate_lon):
                continue
            
            similar_locations.append({
                "image_id": candidate_id,
                "latitude": candidate_lat,
                "longitude": candidate_lon,
                "image_path": candidate_metadata["image_path"],
                "semantic_similarity": float(score),
            })
        
        # Apply distance filtering if we have at least one result
        if similar_locations and min_distance_km > 0:
            # Use first result as reference point for distance filtering
            ref_lat = similar_locations[0]["latitude"]
            ref_lon = similar_locations[0]["longitude"]
            
            filtered_locations = []
            for loc in similar_locations:
                from backend.similar_locations import calculate_distance_km
                distance = calculate_distance_km(ref_lat, ref_lon, loc["latitude"], loc["longitude"])
                loc["distance_km"] = round(distance, 2)
                
                # Keep if distance is significant or it's the first result
                if distance >= min_distance_km or loc == similar_locations[0]:
                    filtered_locations.append(loc)
            
            similar_locations = filtered_locations
        
        # Sort by similarity and take top K
        similar_locations.sort(key=lambda x: x["semantic_similarity"], reverse=True)
        similar_locations = similar_locations[:top_k]
        
        # Add rank
        for rank, location in enumerate(similar_locations, start=1):
            location["rank"] = rank
        
        # Build response
        response = {
            "success": True,
            "uploaded_image": {
                "filename": original_filename,
                "saved_path": str(file_path.relative_to(settings.BASE_DIR)).replace("\\", "/"),
                "file_id": file_id,
            },
            "parameters": {
                "top_k": top_k,
                "min_distance_km": min_distance_km,
            },
            "total_results": len(similar_locations),
            "similar_locations": similar_locations,
        }
        
        return response
    
    except HTTPException:
        raise
    
    except Exception as error:
        if file_path.exists():
            file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Similar location search failed: {str(error)}"
        )



# ============================================================
# IMAGE GALLERY API
# ============================================================

import pandas as pd

@app.get("/api/v1/images")
def get_image_gallery(
    page: int = Query(default=1, ge=1, description="Page number"),
    limit: int = Query(default=20, ge=1, le=100, description="Images per page"),
    search: str = Query(default="", description="Search by image ID (partial match)"),
):
    """Get paginated image gallery"""
    try:
        metadata_path = settings.PROCESSED_METADATA_DIR / "metadata.csv"
        if not metadata_path.exists():
            raise HTTPException(status_code=500, detail="Metadata file not found")
        
        df = pd.read_csv(metadata_path)
        
        if search.strip():
            df = df[df['image_id'].str.contains(search, case=False, na=False)]
        
        total_images = len(df)
        total_pages = (total_images + limit - 1) // limit
        
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        page_df = df.iloc[start_idx:end_idx]
        
        images = []
        for _, row in page_df.iterrows():
            images.append({
                "image_id": row["image_id"],
                "image_url": f"/api/v1/images/{row['image_id']}/file",
                "latitude": float(row["latitude"]) if pd.notna(row["latitude"]) else None,
                "longitude": float(row["longitude"]) if pd.notna(row["longitude"]) else None,
            })
        
        return {
            "success": True,
            "page": page,
            "limit": limit,
            "total_images": total_images,
            "total_pages": total_pages,
            "images": images,
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))



@app.get("/api/v1/images/{image_id}")
def get_image_details(image_id: str):
    """Get detailed metadata for a specific image"""
    try:
        metadata_path = settings.PROCESSED_METADATA_DIR / "metadata.csv"
        df = pd.read_csv(metadata_path)
        matches = df[df['image_id'] == image_id]
        
        if matches.empty:
            raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")
        
        row = matches.iloc[0]
        return {
            "success": True,
            "image": {
                "image_id": row["image_id"],
                "image_url": f"/api/v1/images/{row['image_id']}/file",
                "latitude": float(row["latitude"]) if pd.notna(row["latitude"]) else None,
                "longitude": float(row["longitude"]) if pd.notna(row["longitude"]) else None,
                "image_path": row["image_path"],
            }
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


from fastapi.responses import FileResponse

@app.get("/api/v1/images/{image_id}/file")
def serve_image_file(image_id: str):
    """Serve the actual image file"""
    try:
        if not image_id or ".." in image_id or "/" in image_id:
            raise HTTPException(status_code=400, detail="Invalid image ID")
        
        metadata_path = settings.PROCESSED_METADATA_DIR / "metadata.csv"
        df = pd.read_csv(metadata_path)
        matches = df[df['image_id'] == image_id]
        
        if matches.empty:
            raise HTTPException(status_code=404, detail="Image not found")
        
        image_path_str = matches.iloc[0]["image_path"]
        image_path = settings.BASE_DIR / image_path_str
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found")
        
        return FileResponse(path=str(image_path), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
