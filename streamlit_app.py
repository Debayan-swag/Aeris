import streamlit as st
import requests
from PIL import Image
from io import BytesIO

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="TerraWatch AI",
    page_icon="???",
    layout="wide"
)

def check_server():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.status_code == 200
    except:
        return False

def load_image(image_url):
    try:
        response = requests.get(f"http://localhost:8000{image_url}", timeout=5)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    except:
        pass
    return None

st.title("??? TerraWatch AI")
st.markdown("**Professional Satellite Imagery Discovery Platform**")

with st.sidebar:
    st.title("?? Status")
    if check_server():
        st.success("? API Online")
    else:
        st.error("? API Offline")
        st.stop()
    
    st.divider()
    page = st.radio("Navigation", [
        "?? Home",
        "??? Gallery",
        "?? Search",
        "?? Image Search",
        "?? Locations"
    ])

if page == "?? Home":
    st.header("Welcome")
    st.info("??? **Gallery** - Browse 1,000 satellite images")
    st.info("?? **Search** - Natural language search")
    st.info("?? **Locations** - Find similar terrains")

elif page == "??? Gallery":
    st.header("Gallery")
    
    col1, col2, col3 = st.columns([3,1,1])
    with col1:
        search = st.text_input("Search ID")
    with col2:
        page_num = st.number_input("Page", 1, 100, 1)
    with col3:
        size = st.selectbox("Size", [12,20,30], 0)
    
    params = {"page": page_num, "limit": size}
    if search:
        params["search"] = search
    
    try:
        r = requests.get(f"{API_BASE}/images", params=params)
        data = r.json()
        
        st.success(f"Page {data['page']}/{data['total_pages']} - {data['total_images']} images")
        
        cols = st.columns(4)
        for i, img_data in enumerate(data['images']):
            with cols[i % 4]:
                img = load_image(img_data['image_url'])
                if img:
                    st.image(img, use_container_width=True)
                st.markdown(f"**{img_data['image_id']}**")
                if img_data.get('latitude'):
                    st.caption(f"?? {img_data['latitude']:.3f}, {img_data['longitude']:.3f}")
    except Exception as e:
        st.error(f"Error: {e}")

elif page == "?? Search":
    st.header("Semantic Search")
    
    query = st.text_input("Query", placeholder="urban area")
    top_k = st.slider("Results", 1, 20, 10)
    
    if st.button("Search"):
        if query:
            try:
                r = requests.get(f"{API_BASE}/search/text", params={"query": query, "top_k": top_k})
                data = r.json()
                st.success(f"Found {data['count']} results")
                
                cols = st.columns(3)
                for i, res in enumerate(data['results']):
                    with cols[i % 3]:
                        st.markdown(f"**#{i+1} {res['image_id']}**")
                        img = load_image(res['image_url'])
                        if img:
                            st.image(img, use_container_width=True)
                        st.metric("Score", f"{res['score']:.2%}")
            except Exception as e:
                st.error(f"Error: {e}")

elif page == "?? Image Search":
    st.header("Image-to-Image")
    
    file = st.file_uploader("Upload", type=['jpg','png'])
    top_k = st.slider("Results", 1, 20, 10)
    
    if file and st.button("Find Similar"):
        st.image(Image.open(file), width=300)
        try:
            files = {"file": (file.name, file.getvalue())}
            r = requests.post(f"{API_BASE}/search/image", files=files, params={"top_k": top_k})
            data = r.json()
            st.success(f"Found {data['count']} similar")
            
            cols = st.columns(3)
            for i, res in enumerate(data['results']):
                with cols[i % 3]:
                    st.markdown(f"**#{i+1} {res['image_id']}**")
                    img = load_image(res['image_url'])
                    if img:
                        st.image(img, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

elif page == "?? Locations":
    st.header("Similar Locations")
    
    img_id = st.text_input("Image ID", "S2_000001")
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Results", 1, 20, 10)
    with col2:
        min_dist = st.number_input("Min km", 0.0, 50.0, 1.0)
    
    if st.button("Find"):
        try:
            r = requests.get(f"{API_BASE}/search/similar-locations/{img_id}", 
                           params={"top_k": top_k, "min_distance_km": min_dist})
            data = r.json()
            
            st.subheader("Source")
            src = data['source_location']
            col1, col2 = st.columns([1,2])
            with col1:
                img = load_image(src['image_url'])
                if img:
                    st.image(img, use_container_width=True)
            with col2:
                st.metric("ID", src['image_id'])
                st.metric("Coords", f"{src['latitude']:.4f}, {src['longitude']:.4f}")
            
            st.subheader(f"{data['total_results']} Results")
            for loc in data['similar_locations']:
                with st.expander(f"#{loc['rank']} {loc['image_id']}"):
                    col1, col2 = st.columns([1,2])
                    with col1:
                        img = load_image(loc['image_url'])
                        if img:
                            st.image(img, use_container_width=True)
                    with col2:
                        st.metric("Similarity", f"{loc['semantic_similarity']:.2%}")
                        st.metric("Distance", f"{loc['distance_km']} km")
        except Exception as e:
            st.error(f"Error: {e}")
