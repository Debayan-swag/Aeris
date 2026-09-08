# ?? TerraWatch AI - Quick Start Guide

## What Changed?

**BEFORE:** Search results showed only image IDs (S2_000001, S2_000002...)  
**NOW:** Every search result displays actual satellite image thumbnails!

## Start the System

### Terminal 1: Start Backend
`powershell
cd c:\Users\aritr\OneDrive\Desktop\TerraWatch
uvicorn backend.main:app --reload
`
Wait for: Uvicorn running on http://127.0.0.1:8000

### Terminal 2: Start Streamlit UI
`powershell
cd c:\Users\aritr\OneDrive\Desktop\TerraWatch
streamlit run streamlit_app.py
`
Browser opens automatically at: http://localhost:8501

## Test the Features

### 1. Image Gallery ???
- Click "??? Gallery" in sidebar
- See grid of satellite image thumbnails
- Use page controls to browse 1000 images
- Search for specific IDs (e.g., "S2_0001")

### 2. Semantic Search ??
- Click "?? Search" in sidebar  
- Enter: "urban area with buildings"
- Click Search
- See results with actual satellite images + scores

### 3. Image-to-Image Search ??
- Click "?? Image Search" in sidebar
- Upload any satellite/aerial image
- Click "Find Similar Images"
- See visually similar images with thumbnails

### 4. Similar Locations ??
- Click "?? Locations" in sidebar
- Enter image ID: S2_000001
- Set results: 10, min distance: 1.0 km
- Click "Find Similar Locations"
- See source image + similar locations in different regions

### 5. Click-to-Explore
- From any page, click "?? Explore" on any result
- Automatically jumps to Similar Locations
- Chain explore: Image A ? B ? C ? D

## API Testing

### Run automated tests:
`powershell
python test_visual_gallery.py
`

Expected output:
`
? Health check passed
? Gallery: 1000 images
? Image served: 23620 bytes
? Details: S2_000001
? Search: 10 results
? Similar locations: 10 results

RESULTS: 6/6 tests passed
`

### Manual API tests:

**Gallery:**
`powershell
curl "http://localhost:8000/api/v1/images?page=1&limit=5"
`

**Image Details:**
`powershell
curl "http://localhost:8000/api/v1/images/S2_000001"
`

**Download Image:**
`powershell
curl "http://localhost:8000/api/v1/images/S2_000001/file" -o image.jpg
`

**Search:**
`powershell
curl "http://localhost:8000/api/v1/search/text?query=urban&top_k=5"
`

## Features You Can Now Test

? **Visual Gallery** - Browse all images with thumbnails  
? **Semantic Search** - Text query ? see actual images  
? **Image-to-Image** - Upload ? find similar with thumbnails  
? **Similar Locations** - Find similar terrains elsewhere  
? **Pagination** - Navigate large dataset easily  
? **Click-to-Explore** - Chain searches visually  
? **Search by ID** - Find specific images quickly  

## All Existing Features Still Work

? Semantic Search (OpenCLIP + FAISS)  
? Image-to-Image Search  
? Similar Location Discovery  
? Offline Operation (no internet needed)  
? 1000-image dataset intact  

## Files to Check

- **IMPLEMENTATION_SUMMARY.md** - Complete implementation details
- **VISUAL_GALLERY_UPDATE.md** - Detailed documentation
- **test_visual_gallery.py** - Automated test suite

## Troubleshooting

**Backend won't start:**
- Check if another process uses port 8000
- Solution: 
etstat -ano | findstr :8000 then kill process

**Streamlit shows "API Offline":**
- Start backend first
- Check: http://localhost:8000/api/v1/health

**Images not displaying:**
- Restart backend server
- Check: python test_visual_gallery.py

## Next Steps

1. Test all features in Streamlit UI
2. Try different search queries
3. Upload your own satellite images
4. Explore location chains
5. Check API documentation: http://localhost:8000/docs

---

**Your satellite imagery discovery platform is ready!** ??
