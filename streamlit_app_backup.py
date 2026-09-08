import streamlit as st
import requests
from PIL import Image
import pandas as pd
from pathlib import Path

API_BASE = "http://localhost:8000/api/v1"
PROJECT_ROOT = Path(__file__).parent
IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "images"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata.csv"

st.set_page_config(page_title="TerraWatch AI", page_icon="🛰️", layout="wide")

def check_server():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.status_code == 200
    except:
        return False

def load_metadata():
    if METADATA_PATH.exists():
        return pd.read_csv(METADATA_PATH)
    return None

def display_image(path, caption="", width=None):
    try:
        img_path = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
        if img_path.exists():
            img = Image.open(img_path)
            st.image(img, caption=caption, width=width)
    except Exception as e:
        st.error(f"Error: {e}")

# Main
st.title("🛰️ TerraWatch AI Dashboard")
st.markdown("Satellite Image Semantic Retrieval & Location Discovery")

with st.sidebar:
    st.title("⚙️ System")
    if check_server():
        st.success("✅ Server Online")
    else:
        st.error("❌ Server Offline")
        st.warning("Run: uvicorn backend.main:app --reload")
        st.stop()
    
    st.divider()
    page = st.radio("Navigation", [
        "🏠 Home",
        "🔍 Semantic Search",
        "🖼️ Image Search",
        "📍 Similar Locations",
        "📤 Upload & Find",
        "📊 Dataset"
    ])

if page == "🏠 Home":
    st.header("Welcome to TerraWatch AI")
    st.write("### Available Features:")
    st.info("**🔍 Semantic Search** - Search using natural language")
    st.info("**🖼️ Image Search** - Upload image to find similar ones")
    st.info("**📍 Similar Locations** - Find similar places")
    st.info("**📤 Upload & Find** - Upload any image to find locations")
    st.info("**📊 Dataset** - Explore 1,000 satellite images")

elif page == "🔍 Semantic Search":
    st.header("🔍 Semantic Search")
    query = st.text_input("Search query:", placeholder="urban area with buildings")
    top_k = st.slider("Results:", 1, 20, 5)
    
    if st.button("Search") and query:
        with st.spinner("Searching..."):
            try:
                r = requests.get(f"{API_BASE}/search/text", params={"query": query, "top_k": top_k})
                data = r.json()
                st.success(f"Found {data['count']} results")
                
                cols = st.columns(3)
                for i, res in enumerate(data['results']):
                    with cols[i % 3]:
                        st.write(f"**{i+1}. {res['image_id']}**")
                        st.write(f"Score: {res['score']:.4f}")
                        display_image(res.get('path', ''), width=250)
            except Exception as e:
                st.error(f"Error: {e}")

elif page == "🖼️ Image Search":
    st.header("🖼️ Image-to-Image Search")
    file = st.file_uploader("Upload image:", type=['jpg', 'jpeg', 'png'])
    top_k = st.slider("Results:", 1, 20, 5, key="img")
    
    if file and st.button("Find Similar"):
        with st.spinner("Processing..."):
            try:
                st.subheader("Uploaded")
                st.image(Image.open(file), width=300)
                
                files = {"file": (file.name, file.getvalue())}
                r = requests.post(f"{API_BASE}/search/image", files=files, params={"top_k": top_k})
                data = r.json()
                st.success(f"Found {data['count']} similar images")
                
                cols = st.columns(3)
                for i, res in enumerate(data['results']):
                    with cols[i % 3]:
                        st.write(f"**{i+1}. {res['image_id']}**")
                        st.write(f"Similarity: {res['score']:.4f}")
                        display_image(res.get('path', ''), width=250)
            except Exception as e:
                st.error(f"Error: {e}")

elif page == "📍 Similar Locations":
    st.header("📍 Similar Location Discovery")
    md = load_metadata()
    if md is not None:
        img_id = st.selectbox("Select image:", md['image_id'].tolist()[:100])
        custom = st.text_input("Or enter ID:")
        if custom: img_id = custom
        
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Results:", 1, 20, 5, key="loc")
        with col2:
            min_dist = st.number_input("Min distance (km):", 0.0, 50.0, 1.0)
        
        if st.button("Find Similar Locations"):
            with st.spinner("Searching..."):
                try:
                    r = requests.get(f"{API_BASE}/search/similar-locations/{img_id}",
                                    params={"top_k": top_k, "min_distance_km": min_dist})
                    data = r.json()
                    
                    st.subheader("Source")
                    src = data['source_location']
                    col1, col2 = st.columns([1,2])
                    with col1:
                        display_image(src['image_path'], width=300)
                    with col2:
                        st.write(f"**ID:** {src['image_id']}")
                        st.write(f"**Lat:** {src['latitude']:.6f}")
                        st.write(f"**Lon:** {src['longitude']:.6f}")
                    
                    st.subheader(f"Results ({data['total_results']})")
                    for loc in data['similar_locations']:
                        with st.expander(f"#{loc['rank']} {loc['image_id']} - Score: {loc['semantic_similarity']:.4f}"):
                            col1, col2 = st.columns([1,2])
                            with col1:
                                display_image(loc['image_path'], width=250)
                            with col2:
                                st.metric("Similarity", f"{loc['semantic_similarity']:.4f}")
                                st.metric("Distance", f"{loc['distance_km']} km")
                                st.write(f"Coords: {loc['latitude']:.6f}, {loc['longitude']:.6f}")
                except Exception as e:
                    st.error(f"Error: {e}")

elif page == "📤 Upload & Find":
    st.header("📤 Upload & Find Similar Locations")
    st.info("Upload ANY satellite/aerial image!")
    
    file = st.file_uploader("Upload:", type=['jpg', 'jpeg', 'png'], key="upload")
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Results:", 1, 20, 5, key="upk")
    with col2:
        min_dist = st.number_input("Min distance (km):", 0.0, 50.0, 0.5, key="upd")
    
    if file and st.button("Find Locations"):
        with st.spinner("Analyzing..."):
            try:
                st.subheader("Your Image")
                st.image(Image.open(file), width=400)
                
                files = {"file": (file.name, file.getvalue())}
                r = requests.post(f"{API_BASE}/search/similar-locations-by-image",
                                files=files, params={"top_k": top_k, "min_distance_km": min_dist})
                data = r.json()
                st.success(f"Found {data['total_results']} locations")
                
                for loc in data['similar_locations']:
                    with st.expander(f"#{loc['rank']} {loc['image_id']} - {loc['semantic_similarity']:.4f}"):
                        col1, col2 = st.columns([1,2])
                        with col1:
                            display_image(loc['image_path'], width=250)
                        with col2:
                            st.metric("Similarity", f"{loc['semantic_similarity']:.4f}")
                            if 'distance_km' in loc:
                                st.metric("Distance", f"{loc['distance_km']} km")
                            st.write(f"Coords: {loc['latitude']:.6f}, {loc['longitude']:.6f}")
            except Exception as e:
                st.error(f"Error: {e}")

elif page == "📊 Dataset":
    st.header("📊 Dataset Explorer")
    md = load_metadata()
    if md is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Images", len(md))
        with col2:
            st.metric("Locations", md[['latitude','longitude']].drop_duplicates().shape[0])
        with col3:
            st.metric("Avg Size", f"{md['file_size_bytes'].mean()/1024:.1f} KB")
        with col4:
            st.metric("Format", md['format'].mode()[0])
        
        st.divider()
        st.subheader("Search by ID")
        search = st.text_input("Image ID:", placeholder="S2_000001")
        
        if search:
            res = md[md['image_id'] == search]
            if not res.empty:
                row = res.iloc[0]
                col1, col2 = st.columns([1,2])
                with col1:
                    display_image(row['image_path'], width=350)
                with col2:
                    st.write(f"**ID:** {row['image_id']}")
                    st.write(f"**Name:** {row['image_name']}")
                    st.write(f"**Lat:** {row['latitude']:.6f}")
                    st.write(f"**Lon:** {row['longitude']:.6f}")
                    st.write(f"**Size:** {row['width']}x{row['height']}")
                    st.write(f"**Format:** {row['format']}")
                    st.write(f"**File:** {row['file_size_bytes']/1024:.2f} KB")
        
        st.divider()
        st.subheader("Browse Dataset")
        rows = st.selectbox("Rows:", [10, 25, 50, 100])
        st.dataframe(md.head(rows), use_container_width=True)
        
        csv = md.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "metadata.csv", "text/csv")
