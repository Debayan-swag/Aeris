# Test Script for Visual Gallery System

import requests
import sys

API_BASE = "http://localhost:8000/api/v1"

def test_health():
    print("\n1. Testing Health Check...")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        if r.status_code == 200:
            print("   ? Health check passed")
            return True
        else:
            print(f"   ? Health check failed: {r.status_code}")
            return False
    except Exception as e:
        print(f"   ? Server not running: {e}")
        print("   Start server: uvicorn backend.main:app --reload")
        return False

def test_gallery():
    print("\n2. Testing Image Gallery...")
    try:
        r = requests.get(f"{API_BASE}/images", params={"page": 1, "limit": 5})
        data = r.json()
        if data['success'] and len(data['images']) > 0:
            print(f"   ? Gallery: {data['total_images']} images, page {data['page']}/{data['total_pages']}")
            print(f"   Sample image: {data['images'][0]['image_id']}")
            print(f"   Image URL: {data['images'][0]['image_url']}")
            return True
        else:
            print("   ? Gallery returned empty")
            return False
    except Exception as e:
        print(f"   ? Gallery test failed: {e}")
        return False

def test_image_serve():
    print("\n3. Testing Image Serving...")
    try:
        r = requests.get(f"{API_BASE}/images/S2_000001/file")
        if r.status_code == 200 and r.headers.get('content-type') == 'image/jpeg':
            print(f"   ? Image served: {len(r.content)} bytes")
            return True
        else:
            print(f"   ? Image serve failed: {r.status_code}")
            return False
    except Exception as e:
        print(f"   ? Image serve failed: {e}")
        return False

def test_image_details():
    print("\n4. Testing Image Details...")
    try:
        r = requests.get(f"{API_BASE}/images/S2_000001")
        data = r.json()
        if data['success']:
            img = data['image']
            print(f"   ? Details: {img['image_id']}")
            print(f"   Coords: {img['latitude']:.4f}, {img['longitude']:.4f}")
            print(f"   URL: {img['image_url']}")
            return True
        else:
            print("   ? Details failed")
            return False
    except Exception as e:
        print(f"   ? Details failed: {e}")
        return False

def test_semantic_search():
    print("\n5. Testing Semantic Search with Image URLs...")
    try:
        r = requests.get(f"{API_BASE}/search/text", params={"query": "urban area", "top_k": 3})
        data = r.json()
        if data['count'] > 0 and 'image_url' in data['results'][0]:
            print(f"   ? Search: {data['count']} results")
            print(f"   Result 1: {data['results'][0]['image_id']}")
            print(f"   Image URL: {data['results'][0]['image_url']}")
            print(f"   Score: {data['results'][0]['score']:.4f}")
            return True
        else:
            print("   ? Search failed or missing image_url")
            return False
    except Exception as e:
        print(f"   ? Search failed: {e}")
        return False

def test_similar_locations():
    print("\n6. Testing Similar Locations with Image URLs...")
    try:
        r = requests.get(f"{API_BASE}/search/similar-locations/S2_000001", 
                        params={"top_k": 3, "min_distance_km": 1.0})
        data = r.json()
        if data['success'] and len(data['similar_locations']) > 0:
            print(f"   ? Similar locations: {data['total_results']} results")
            print(f"   Source: {data['source_location']['image_id']}")
            print(f"   Source URL: {data['source_location']['image_url']}")
            loc = data['similar_locations'][0]
            print(f"   Result 1: {loc['image_id']} ({loc['semantic_similarity']:.2%})")
            print(f"   Result URL: {loc['image_url']}")
            return True
        else:
            print("   ? Similar locations failed")
            return False
    except Exception as e:
        print(f"   ? Similar locations failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("TERRAWATCH AI - VISUAL GALLERY SYSTEM TEST")
    print("="*60)
    
    if not test_health():
        print("\n? Server not running. Start with:")
        print("   uvicorn backend.main:app --reload")
        sys.exit(1)
    
    results = [
        test_gallery(),
        test_image_serve(),
        test_image_details(),
        test_semantic_search(),
        test_similar_locations(),
    ]
    
    print("\n" + "="*60)
    print(f"RESULTS: {sum(results)}/6 tests passed")
    print("="*60)
    
    if all(results):
        print("\n? ALL TESTS PASSED!")
        print("\nNext steps:")
        print("1. Start Streamlit: streamlit run streamlit_app.py")
        print("2. Open browser to view visual gallery")
        print("3. Browse images, search, and explore locations")
    else:
        print("\n? Some tests failed")
        sys.exit(1)
