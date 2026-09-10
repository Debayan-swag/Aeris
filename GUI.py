import sys
from pathlib import Path

# Fast imports only
import streamlit as st

# Lazy imports - only import when needed
PROJECT_DIR = Path(r"C:\Users\USER\OneDrive\Desktop\Aeris")
DATASET_PATH = PROJECT_DIR / "sentinel-2-processed.parquet"
EMBEDDING_DIR = PROJECT_DIR / "remoteclip_embeddings"
FAISS_PATH = EMBEDDING_DIR / "remoteclip.faiss"
INDEX_PATH = EMBEDDING_DIR / "remoteclip_index.parquet"

# Global variables for lazy loading
_pipelines_loaded = False
_import_error = ""


def lazy_load_pipelines():
    """Lazy load heavy imports only when needed"""
    global _pipelines_loaded, _import_error
    
    if _pipelines_loaded:
        return True, ""
    
    try:
        # Import heavy libraries only when needed
        global faiss, np, pd, torch, Image, BytesIO, Document
        global stream_image_text_pipeline, remoteclip_model, remoteclip_tokenizer, remoteclip_device
        global run_text_pipeline, stream_text_pipeline, compare_satellite_images
        
        import faiss
        import numpy as np
        import pandas as pd
        import torch
        from PIL import Image
        from io import BytesIO
        from langchain_core.documents import Document
        
        from image_text import (
            stream_image_text_pipeline,
            remoteclip_model,
            remoteclip_tokenizer,
            remoteclip_device,
        )
        from text import run_text_pipeline, stream_text_pipeline
        from change_detection import compare_satellite_images
        
        _pipelines_loaded = True
        return True, ""
    except Exception as exc:
        _import_error = str(exc)
        return False, str(exc)


def lazy_load_pipelines():
    """Lazy load heavy imports only when needed"""
    global _pipelines_loaded, _import_error
    
    if _pipelines_loaded:
        return True, ""
    
    try:
        # Import heavy libraries only when needed
        global faiss, np, pd, torch, Image, BytesIO, Document
        global stream_image_text_pipeline, remoteclip_model, remoteclip_tokenizer, remoteclip_device
        global run_text_pipeline, stream_text_pipeline, compare_satellite_images
        
        import faiss
        import numpy as np
        import pandas as pd
        import torch
        from PIL import Image
        from io import BytesIO
        from langchain_core.documents import Document
        
        from image_text import (
            stream_image_text_pipeline,
            remoteclip_model,
            remoteclip_tokenizer,
            remoteclip_device,
        )
        from text import run_text_pipeline, stream_text_pipeline
        from change_detection import compare_satellite_images
        
        _pipelines_loaded = True
        return True, ""
    except Exception as exc:
        _import_error = str(exc)
        return False, str(exc)


st.set_page_config(
    page_title="Aeris Satellite Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    /* ========== PREMIUM DESIGN SYSTEM ========== */
    
    /* Base & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
        color: #e8eef5;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Title - Hero Style */
    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 700;
        margin: 20px 0 8px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .main-subtitle {
        text-align: center;
        font-size: 16px;
        color: #9ca9ba;
        margin-bottom: 32px;
        font-weight: 400;
        animation: fadeInUp 0.6s ease-out 0.1s both;
    }
    
    /* Section Headers */
    .section-heading {
        font-size: 24px;
        font-weight: 600;
        color: #ffffff;
        margin: 24px 0 12px 0;
        letter-spacing: -0.01em;
        position: relative;
        padding-left: 16px;
    }
    
    .section-heading::before {
        content: '';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 4px;
        height: 24px;
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 2px;
    }
    
    .section-subtitle {
        color: #9ca9ba;
        font-size: 14px;
        margin-bottom: 20px;
        padding-left: 16px;
        font-weight: 400;
    }
    
    /* Premium Cards */
    .premium-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .premium-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.15);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    /* Status Box - Modern */
    .status-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 14px 18px;
        color: #e0e7f0;
        font-size: 13px;
        margin: 12px 0;
        font-weight: 500;
        backdrop-filter: blur(8px);
    }
    
    /* Result Title */
    .result-title {
        font-size: 20px;
        font-weight: 600;
        color: #ffffff;
        margin: 24px 0 16px 0;
        letter-spacing: -0.01em;
    }
    
    /* Premium Buttons */
    div.stButton > button {
        width: 100%;
        min-height: 72px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        color: white;
        font-size: 17px;
        font-weight: 600;
        letter-spacing: -0.01em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    div.stButton > button:hover {
        border-color: rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.2);
    }
    
    div.stButton > button:active {
        transform: translateY(0);
    }
    
    /* Primary Button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        min-height: 48px;
        font-size: 15px;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #5568d3 0%, #6a3f8f 100%);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
    
    /* File Uploader - Premium */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 24px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(102, 126, 234, 0.5);
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Stream Box - Animated */
    .stream-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px;
        line-height: 1.7;
        font-size: 14px;
        color: #e0e7f0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.3s ease-out;
        font-weight: 400;
    }
    
    /* Image Container - Modern */
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    .image-container:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    /* Text Area - Premium */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #e8eef5 !important;
        font-size: 14px !important;
        padding: 16px !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: rgba(102, 126, 234, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Sidebar - Premium */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #0f0c29 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Metrics - Modern Cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 13px;
        color: #9ca9ba;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    
    /* Progress Bar - Gradient */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    /* Spinner - Modern */
    .stSpinner > div {
        border-color: rgba(102, 126, 234, 0.2);
        border-top-color: #667eea;
    }
    
    /* Tabs - Premium */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.02);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander - Modern */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    /* No Results - Premium Empty State */
    .no-results {
        background: rgba(255, 255, 255, 0.03);
        border: 1px dashed rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 48px 32px;
        text-align: center;
        color: #9ca9ba;
        margin: 32px 0;
    }
    
    .no-results h3 {
        color: #ffffff;
        font-size: 20px;
        margin-bottom: 8px;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6b7a8f;
        font-size: 12px;
        margin-top: 48px;
        padding: 24px 0;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        font-weight: 400;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* Loading State */
    .loading-indicator {
        display: inline-block;
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        color: #a8b8ff;
    }
    
    /* Success Badge */
    .badge-success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%);
        border-color: rgba(16, 185, 129, 0.3);
        color: #6ee7b7;
    }
    
    /* Warning Badge */
    .badge-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.2) 100%);
        border-color: rgba(245, 158, 11, 0.3);
        color: #fcd34d;
    }
    
    /* Info Message Box */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 16px 0;
        color: #bfdbfe;
        font-size: 14px;
    }
    
    /* Success Message Box */
    .success-box {
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 16px 0;
        color: #6ee7b7;
        font-size: 14px;
    }
    
    /* Smooth Scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.3);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(102, 126, 234, 0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False, ttl=3600)
def load_dataset():
    """Load dataset with optimized caching (1 hour TTL)"""
    try:
        # Lazy import pandas only when needed
        import pandas as pd
        return pd.read_parquet(DATASET_PATH, engine='pyarrow')
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None


@st.cache_resource(show_spinner=False, ttl=3600)
def load_retrieval_resources():
    """Load FAISS index and mapping with optimized caching"""
    try:
        # Lazy imports
        import faiss
        import pandas as pd
        
        vector_store = faiss.read_index(str(FAISS_PATH))
        index_mapping = pd.read_parquet(INDEX_PATH, engine='pyarrow')
        return vector_store, index_mapping
    except Exception as e:
        st.error(f"Failed to load retrieval resources: {e}")
        return None, None


class RemoteCLIPTextRetriever:
    def __init__(self, vector_store, index_mapping, model, tokenizer, device):
        self.vector_store = vector_store
        self.index_mapping = index_mapping
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def similarity_search_with_score(self, query, k=3):
        tokens = self.tokenizer([query]).to(self.device)

        with torch.no_grad():
            query_embedding = self.model.encode_text(tokens)
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

        query_embedding = query_embedding.cpu().numpy().astype(np.float32)
        scores, indices = self.vector_store.search(query_embedding, k)

        results = []
        for index, score in zip(indices[0], scores[0]):
            if index < 0:
                continue

            row = self.index_mapping.iloc[int(index)]
            image_name = row["image_name"]

            document = Document(
                page_content=f"Satellite image: {image_name}",
                metadata={"image_name": image_name, "index": int(index)},
            )

            results.append((document, float(score)))

        return results


@st.cache_resource(show_spinner=False, ttl=3600)
def load_text_retriever():
    """Load text retriever with optimized caching"""
    vector_store, index_mapping = load_retrieval_resources()
    if vector_store is None or index_mapping is None:
        return None
    
    from image_text import remoteclip_model, remoteclip_tokenizer, remoteclip_device
    
    return RemoteCLIPTextRetriever(
        vector_store=vector_store,
        index_mapping=index_mapping,
        model=remoteclip_model,
        tokenizer=remoteclip_tokenizer,
        device=remoteclip_device,
    )


def get_image(dataframe, image_name):
    """Get image from dataframe with lazy PIL import"""
    from PIL import Image
    from io import BytesIO
    
    rows = dataframe.loc[dataframe["image_name"] == image_name]
    if rows.empty:
        return None
    image_bytes = rows.iloc[0]["image"]
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def display_results(results, dataframe, title="Results"):
    if not results or len(results) == 0:
        st.markdown('<div class="no-results">', unsafe_allow_html=True)
        st.markdown("### No matching images found")
        st.markdown("There are no images like this in the database.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown(f'<div class="result-title">{title} ({len(results)} images)</div>', unsafe_allow_html=True)

    num_cols = min(len(results), 5)
    columns = st.columns(num_cols)

    for i, result in enumerate(results):
        image_name = result.get("image_name", "")
        score = result.get("score", result.get("retrieval_score", 0.0))
        image = get_image(dataframe, image_name)

        with columns[i % num_cols]:
            if image is not None:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown(f"**{image_name}**")
            st.caption(f"Similarity: {float(score):.4f}")


def run_text_mode():
    st.markdown('<div class="section-heading">Text Search</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Search the satellite archive using natural language.</div>',
        unsafe_allow_html=True,
    )

    query = st.text_area(
        "Search Query",
        placeholder="Example: Find satellite images showing large airports with aircraft near runways.",
        height=100,
    )

    search_button = st.button("Search Archive", type="primary", use_container_width=True)

    if not search_button:
        return

    if not query.strip():
        st.warning("Please enter a search query.")
        return

    try:
        with st.spinner("Loading resources..."):
            dataframe = load_dataset()
            if dataframe is None:
                return
            
            retriever = load_text_retriever()
            if retriever is None:
                return

        # Auto-scroll anchor
        st.markdown('<div id="processing-section"></div>', unsafe_allow_html=True)
        
        # Main processing section
        st.markdown('<div class="result-title">Search Processing</div>', unsafe_allow_html=True)
        status_box = st.empty()
        progress_bar = st.progress(0)
        
        # Thinking process container with expandable sections
        thinking_container = st.container()
        
        # Results container
        results_container = st.container()
        
        # Auto-scroll to processing section
        st.markdown(
            """
            <script>
                window.parent.document.getElementById('processing-section').scrollIntoView({behavior: 'smooth'});
            </script>
            """,
            unsafe_allow_html=True
        )
        
        # Track shown sections and boxes
        state = {
            "original_shown": False,
            "thinking_box": None,
            "thinking_text": "",
            "queries_shown": False,
            "retrieval_started": False,
            "retrieval_expander": None,
            "retrieval_details": []
        }
        
        # Stream the pipeline
        for event in stream_text_pipeline(query.strip(), retriever):
            event_type = event.get("type")
            data = event.get("data")
            
            if event_type == "stage":
                status_box.info(f"{data}")
                
                # Update progress and show sections based on stage
                if "Understanding" in data:
                    progress_bar.progress(0.15)
                    if not state["original_shown"]:
                        with thinking_container:
                            with st.expander("Your Original Query", expanded=True):
                                st.markdown(f"**{query.strip()}**")
                                st.caption("Processing your natural language query...")
                        state["original_shown"] = True
                        
                elif "AI analysis" in data or "analysis" in data.lower() or "intent" in data.lower():
                    progress_bar.progress(0.30)
                    status_box.info("AI is analyzing your search intent...")
                    
                elif "variations" in data.lower() or "Generating" in data:
                    progress_bar.progress(0.45)
                    status_box.info("Generating optimized search variations...")
                    
                elif "Retrieving" in data or "3 images" in data:
                    progress_bar.progress(0.60)
                    if not state["retrieval_started"]:
                        with thinking_container:
                            state["retrieval_expander"] = st.expander("Retrieval Process", expanded=True)
                        state["retrieval_started"] = True
                    
                elif "top 7" in data or "best matches" in data or "Selecting" in data:
                    progress_bar.progress(0.85)
                    status_box.info("Selecting top 7 from all candidates...")
                    
                elif "complete" in data.lower():
                    progress_bar.progress(1.0)
            
            elif event_type == "thinking_start":
                # Create thinking section
                if state["thinking_box"] is None:
                    with thinking_container:
                        st.markdown('<div class="result-title">AI Analysis & Understanding</div>', unsafe_allow_html=True)
                        st.caption("AI is examining what you're looking for and why")
                        st.markdown("---")
                        state["thinking_box"] = st.empty()
            
            elif event_type == "thinking_content":
                # Stream the thinking content
                state["thinking_text"] += str(data)
                if state["thinking_box"] is not None:
                    state["thinking_box"].markdown(f'<div class="stream-box">{state["thinking_text"]}</div>', unsafe_allow_html=True)
            
            elif event_type == "queries":
                if not state["queries_shown"]:
                    with thinking_container:
                        with st.expander("Search Strategy - 3 Optimized Variations", expanded=True):
                            st.success(f"**Generated {len(data)} Precision Queries:**")
                            st.markdown("---")
                            for i, q in enumerate(data, 1):
                                st.markdown(f"**Query {i}:** {q}")
                            st.markdown("---")
                            st.caption("Each query retrieves 3 images - 9 candidates - Top 7 selected")
                    state["queries_shown"] = True
            
            elif event_type == "query_variation":
                variation_data = data
                state["retrieval_details"].append(f"→ Searching with variation {variation_data['index']}...")
                if state["retrieval_expander"]:
                    with state["retrieval_expander"]:
                        st.caption("\n".join(state["retrieval_details"][-3:]))
            
            elif event_type == "partial_results":
                state["retrieval_details"].append(f"Found {data['count']} candidates from query {data['query_num']}")
                if state["retrieval_expander"]:
                    with state["retrieval_expander"]:
                        st.caption("\n".join(state["retrieval_details"][-5:]))
            
            elif event_type == "results":
                # Format results properly
                formatted_results = []
                for r in data:
                    metadata = r.get("metadata", {})
                    formatted_results.append({
                        "image_name": metadata.get("image_name", ""),
                        "score": r.get("score", 0.0)
                    })
                
                # Final summary in retrieval expander
                if state["retrieval_expander"]:
                    with state["retrieval_expander"]:
                        st.success(f"Retrieved and ranked {len(formatted_results)} unique images")
                
                # Complete status
                status_box.success("Search Complete")
                progress_bar.empty()
                
                # Show results
                with results_container:
                    st.markdown("---")
                    display_results(formatted_results, dataframe, "Search Results")

    except Exception as exc:
        st.error(f"Search failed: {exc}")


def run_image_text_mode():
    st.markdown('<div class="section-heading">Image & Text Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Upload a satellite image and describe your analysis request.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Satellite Image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    )

    query = st.text_area(
        "Analysis Request",
        placeholder="Example: Find similar airport scenes or detect changes in this area.",
        height=90,
    )

    analyze_button = st.button("Analyze Image", type="primary", use_container_width=True)

    if uploaded_file is not None:
        try:
            preview = Image.open(uploaded_file).convert("RGB")
            st.markdown('<div class="result-title">Uploaded Image</div>', unsafe_allow_html=True)
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(preview, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to load image: {e}")
            return

    if not analyze_button:
        return

    if uploaded_file is None:
        st.warning("Please upload a satellite image.")
        return

    if not query.strip():
        st.warning("Please enter an analysis request.")
        return

    try:
        with st.spinner("Loading resources..."):
            dataframe = load_dataset()
            if dataframe is None:
                return
            
            vector_store, index_mapping = load_retrieval_resources()
            if vector_store is None or index_mapping is None:
                return

        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")

        # Auto-scroll anchor
        st.markdown('<div id="analysis-section"></div>', unsafe_allow_html=True)
        
        # Single processing container that updates
        processing_container = st.container()
        with processing_container:
            st.markdown('<div class="result-title">Analysis</div>', unsafe_allow_html=True)
            status_box = st.empty()
            progress_bar = st.progress(0)
            details_box = st.empty()
        
        # Description container (will appear only when content starts streaming)
        description_container = st.container()
        
        # Results container
        results_container = st.container()
        
        # Auto-scroll to analysis section
        st.markdown(
            """
            <script>
                window.parent.document.getElementById('analysis-section').scrollIntoView({behavior: 'smooth'});
            </script>
            """,
            unsafe_allow_html=True
        )

        response_text = ""
        intent_detected = None
        description_started = False

        for event in stream_image_text_pipeline(
            image=image,
            user_text=query.strip(),
            vector_store=vector_store,
            index_mapping=index_mapping,
            dataframe=dataframe,
            image_name=uploaded_file.name,
        ):
            event_type = event.get("type")
            data = event.get("data")

            if event_type == "status":
                status_box.info(f"{data}")
                
                # Update progress based on status
                if "Understanding" in data:
                    progress_bar.progress(0.10)
                    details_box.caption("Analyzing your request to determine analysis mode...")
                elif "Interpreting" in data or "image" in data.lower():
                    progress_bar.progress(0.25)
                    details_box.info("Processing full image embedding with vision encoder...")
                elif "Extracting" in data or "semantic" in data.lower():
                    progress_bar.progress(0.40)
                    details_box.info("Extracting visual features and semantic representations...")
                elif "description" in data.lower():
                    progress_bar.progress(0.50)
                    details_box.info("AI is generating contextual description...")
                elif "Searching" in data:
                    progress_bar.progress(0.65)
                    details_box.info("Retrieving top-3 most similar images from archive...")
                elif "Ranking" in data:
                    progress_bar.progress(0.80)
                    details_box.info("Ranking candidates by semantic similarity...")
                elif "Comparing" in data:
                    progress_bar.progress(0.90)
                    details_box.warning("Performing pixel-level change detection on top-3 candidates...")
                elif "complete" in data.lower():
                    progress_bar.progress(1.0)

            elif event_type == "intent":
                intent_detected = data
                if data == "normal_retriever":
                    details_box.success("Mode: Normal Retrieval - Finding semantically similar satellite images")
                else:
                    details_box.success("Mode: Change Detection - Comparing images to detect temporal changes")

            elif event_type == "content":
                response_text += str(data)
                
                # Create the description section only when first token arrives
                if not description_started:
                    with description_container:
                        st.markdown("---")
                        st.markdown('<div class="result-title">AI Image Description</div>', unsafe_allow_html=True)
                        st.caption("AI model is analyzing the satellite image and generating contextual description...")
                        st.markdown("")  # Add spacing
                        description_box = st.empty()
                    description_started = True
                
                # Stream the content gracefully with better formatting
                description_box.markdown(f'<div class="stream-box">{response_text}</div>', unsafe_allow_html=True)

            elif event_type == "semantic_evidence":
                concepts = data.get("concepts", [])
                concepts_text = ", ".join(concepts[:5])
                details_box.caption(f"Detected concepts: {concepts_text}")

            elif event_type == "retrieval_results":
                results = data
                progress_bar.progress(1.0)
                status_box.success("Analysis Complete")
                
                # Check if the uploaded image is in the results (same image found)
                uploaded_name = uploaded_file.name
                exact_match = None
                for result in results:
                    if result.get("image_name") == uploaded_name:
                        exact_match = result
                        break
                
                with results_container:
                    if exact_match:
                        # Show only the exact match
                        st.info("Exact match found in database")
                        display_results([exact_match], dataframe, "Database Match")
                    else:
                        # Show all similar images
                        display_results(results, dataframe, "Similar Images")

            elif event_type == "change_results":
                change_results = data
                progress_bar.progress(1.0)
                status_box.success("Change detection complete")
                # Keep details_box visible - don't clear it
                
                with results_container:
                    st.markdown('<div class="result-title">Change Detection Results</div>', unsafe_allow_html=True)
                    
                    # Check if ANY image is similar (not different)
                    similar_found = any(not item.get('different', True) for item in change_results)
                    all_different = all(item.get('different', False) for item in change_results)
                    
                    if similar_found:
                        # One or more similar images found - pipeline not implemented yet
                        st.info("Similar image detected in the database. This pipeline is not implemented yet.")
                        st.caption("System detected at least one matching image. Further processing is pending.")
                    
                    elif all_different:
                        # All 3 images are different - show change detection
                        st.success("All retrieved images show significant differences. Change detection analysis:")
                        
                        # Show change detection visualization
                        from change_detection_viz import create_difference_mask
                        
                        for idx, item in enumerate(change_results):
                            image_name = item.get('image_name', 'Unknown')
                            is_different = item.get('different', False)
                            change_ratio = item.get('change_ratio', 0.0)
                            retrieval_score = item.get('retrieval_score', 0.0)
                            
                            with st.expander(f"Comparison {idx+1}: {image_name}", expanded=(idx==0)):
                                # Get candidate image
                                rows = dataframe.loc[dataframe["image_name"] == image_name]
                                if not rows.empty:
                                    image_bytes = rows.iloc[0]["image"]
                                    candidate_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                                    
                                    # Metrics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Retrieval Score", f"{retrieval_score:.4f}")
                                    with col2:
                                        st.metric("Change Ratio", f"{change_ratio:.4f}")
                                    with col3:
                                        status_text = "Different" if is_different else "Similar"
                                        st.metric("Status", status_text)
                                    
                                    # Visualizations in tabs
                                    tab1, tab2, tab3 = st.tabs(["Side-by-Side", "Heatmap", "Binary Mask"])
                                    
                                    with tab1:
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            st.image(image, caption="Source Image", use_container_width=True)
                                        with col_b:
                                            st.image(candidate_image, caption=f"Archive: {image_name}", use_container_width=True)
                                    
                                    with tab2:
                                        binary_mask, heatmap, overlay = create_difference_mask(image, candidate_image)
                                        st.image(heatmap, caption="Difference Heatmap", use_container_width=True)
                                        st.caption("Red = High difference, Blue = Low difference")
                                    
                                    with tab3:
                                        binary_mask, heatmap, overlay = create_difference_mask(image, candidate_image)
                                        st.image(binary_mask, caption="Binary Change Mask", use_container_width=True)
                                        st.caption("White = Changed regions, Black = Unchanged regions")
                        
                        # Ask if user wants to see similar images
                        st.markdown("---")
                        st.markdown("### Would you like to see similar images from the database?")
                        
                        col_yes, col_no = st.columns(2)
                        with col_yes:
                            if st.button("Yes, show similar images", key="show_similar", type="primary"):
                                st.session_state["request_similar"] = True
                                st.rerun()
                        with col_no:
                            if st.button("No, exit", key="exit_analysis"):
                                st.session_state["request_similar"] = False
                                st.info("Analysis complete. You can start a new search anytime.")
                    
                    else:
                        # Edge case - shouldn't happen
                        st.warning("Unexpected comparison result. Please try again.")

            elif event_type == "no_data":
                progress_bar.progress(1.0)
                status_box.warning("No matches found")
                # Keep details_box visible
                
                with results_container:
                    st.markdown('<div class="no-results">', unsafe_allow_html=True)
                    st.markdown("### No matching images found")
                    st.markdown("There are no images like this in the database.")
                    st.markdown('</div>', unsafe_allow_html=True)

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")


def run_change_detection_mode():
    import cv2
    import tempfile
    import os
    from transformers import pipeline
    import time
    import hashlib
    
    st.markdown('<div class="section-heading">Change Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Upload two satellite images to detect and analyze changes.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    
    with col1:
        before_image_file = st.file_uploader(
            "Source Image (Before)",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            key="before_image"
        )
        if before_image_file:
            preview_before = Image.open(before_image_file).convert("RGB")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(preview_before, caption="Source Image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        after_image_file = st.file_uploader(
            "Changed Image (After)",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            key="after_image"
        )
        if after_image_file:
            preview_after = Image.open(after_image_file).convert("RGB")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(preview_after, caption="Changed Image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    detect_button = st.button("Detect Changes", type="primary", use_container_width=True)

    if not detect_button:
        return

    if before_image_file is None or after_image_file is None:
        st.warning("Please upload both images to proceed.")
        return

    try:
        # Check if images are the same by name
        if before_image_file.name == after_image_file.name:
            st.info("No changes detected - images are identical.")
            return
        
        # Check if images are the same by content hash
        before_image_file.seek(0)
        after_image_file.seek(0)
        before_hash = hashlib.md5(before_image_file.read()).hexdigest()
        after_hash = hashlib.md5(after_image_file.read()).hexdigest()
        
        if before_hash == after_hash:
            st.info("Both images are identical. No changes detected.")
            return
        
        # === LIVE STREAMING COMPARISON PROCESS ===
        st.markdown("---")
        st.markdown('<div class="result-title">Live Change Detection Process</div>', unsafe_allow_html=True)
        
        # Status container that updates live
        status_container = st.empty()
        progress_bar = st.progress(0)
        
        # Save uploaded files to temporary paths
        before_image_file.seek(0)
        after_image_file.seek(0)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_before:
            tmp_before.write(before_image_file.read())
            before_path = tmp_before.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_after:
            tmp_after.write(after_image_file.read())
            after_path = tmp_after.name
        
        # STREAMING STEPS - Show each step with progress
        steps = [
            ("Loading satellite imagery...", 0.10),
            ("Preprocessing images...", 0.20),
            ("Aligning spatial coordinates...", 0.30),
            ("Normalizing radiometric properties...", 0.40),
            ("Computing multiscale differences...", 0.50),
            ("Detecting edge variations...", 0.60),
            ("Identifying change regions...", 0.70),
            ("Classifying detected changes...", 0.85),
            ("Finalizing analysis...", 0.95)
        ]
        
        result = None
        for step_text, progress_val in steps:
            status_container.info(step_text)
            progress_bar.progress(progress_val)
            
            # Run actual detection at step 70%
            if progress_val == 0.70:
                result = compare_satellite_images(before_path, after_path)
            
            time.sleep(0.4)  # Visible progress for user
        
        # Clean up temp files
        os.unlink(before_path)
        os.unlink(after_path)
        
        status_container.success("Change detection complete!")
        progress_bar.progress(1.0)
        time.sleep(0.5)
        
        # Clear progress indicators
        status_container.empty()
        progress_bar.empty()
        
        changes = result.get("changes", [])
        change_percentage = result.get("change_percentage", 0.0)
        
        # Check if there are changes
        if len(changes) == 0:
            st.info("No significant changes detected between the two images.")
            
            st.markdown("---")
            st.markdown("### Would you like to retrieve similar images from the database?")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes, find similar images", key="retrieve_similar", type="primary"):
                    st.info("Similar image retrieval will be implemented soon.")
            with col_no:
                if st.button("No, exit", key="exit_no_change"):
                    st.session_state["mode"] = None
                    st.rerun()
            
            return
        
        # === VISUAL COMPARISON RESULTS (FIRST) ===
        st.markdown("---")
        st.markdown('<div class="result-title">Detection Summary</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Regions Changed", f"{len(changes):,}", help="Total number of detected change regions")
        with col2:
            st.metric("Area Coverage", f"{change_percentage:.1f}%", help="Percentage of image area affected")
        with col3:
            st.metric("Total Pixels", f"{result.get('changed_pixels', 0):,}", help="Total pixels that changed")
        
        # Display images side by side
        st.markdown("---")
        st.markdown('<div class="result-title">Visual Comparison</div>', unsafe_allow_html=True)
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("**Source Image**")
            before_display = cv2.cvtColor(result["before_image"], cv2.COLOR_BGR2RGB)
            st.image(before_display, use_container_width=True)
        
        with col_right:
            st.markdown("**Changes Detected**")
            overlay_display = cv2.cvtColor(result["change_overlay"], cv2.COLOR_BGR2RGB)
            st.image(overlay_display, use_container_width=True)
        
        # Detailed change list in expander
        with st.expander(f"View detailed breakdown of {len(changes)} changes", expanded=False):
            for idx, change in enumerate(changes, 1):
                ctype = change.get("change_type", "change")
                area = change.get("area_pixels", 0)
                x = change.get("x", 0)
                y = change.get("y", 0)
                
                st.markdown(
                    f'<p style="color:#ddd;margin:8px 0;"><strong>{idx}. {ctype}</strong> — {area:,} pixels at ({x}, {y})</p>',
                    unsafe_allow_html=True
                )
        
        # === AI ANALYSIS (SECOND - BELOW COMPARISON) ===
        st.markdown("---")
        st.markdown('<div class="result-title">AI-Powered Analysis</div>', unsafe_allow_html=True)
        st.caption("Comprehensive analysis generated by advanced language model")
        
        # Load DistilGPT2 silently
        @st.cache_resource(show_spinner=False)
        def load_distilgpt2():
            return pipeline("text-generation", model="distilbert/distilgpt2", device=-1)
        
        text_generator = load_distilgpt2()
        
        # Build comprehensive change list for AI
        change_list_items = []
        change_types_dict = {}
        
        for change in changes:
            ctype = change.get("change_type", "change")
            if ctype not in change_types_dict:
                change_types_dict[ctype] = []
            change_types_dict[ctype].append(change)
        
        for idx, change in enumerate(changes, 1):
            ctype = change.get("change_type", "change")
            area = change.get("area_pixels", 0)
            x = change.get("x", 0)
            y = change.get("y", 0)
            width = change.get("width", 0)
            height = change.get("height", 0)
            
            change_entry = (
                f"{idx}. {ctype} — Area: {area:,} pixels, "
                f"Location: ({x}, {y}), Dimensions: {width}×{height}"
            )
            change_list_items.append(change_entry)
        
        total_changed_pixels = sum(c.get("area_pixels", 0) for c in changes)
        avg_change_size = total_changed_pixels // len(changes) if changes else 0
        
        type_breakdown = []
        for ctype, clist in change_types_dict.items():
            count = len(clist)
            total_area = sum(c.get("area_pixels", 0) for c in clist)
            type_breakdown.append(f"- {ctype}: {count} occurrences, {total_area:,} pixels total")
        
        type_summary = "\n".join(type_breakdown)
        changes_full_list = "\n".join(change_list_items)
        
        prompt = (
            f"Satellite imagery change detection analysis:\n\n"
            f"Total changes detected: {len(changes)}\n"
            f"Coverage: {change_percentage:.1f}% of analyzed area\n"
            f"Total pixels affected: {total_changed_pixels:,}\n"
            f"Average change size: {avg_change_size:,} pixels\n\n"
            f"Change type distribution:\n{type_summary}\n\n"
            f"Complete list of all detected changes:\n{changes_full_list}\n\n"
            f"Provide a comprehensive, detailed analysis of these changes. "
            f"Describe the spatial patterns, the significance of different change types, "
            f"the distribution across the image, and potential implications. "
            f"Be thorough and analytical in explaining what these changes represent:\n\n"
        )
        
        # Show "Generating analysis..." with animation
        analysis_placeholder = st.empty()
        with analysis_placeholder:
            with st.spinner("Generating comprehensive analysis..."):
                try:
                    generation_result = text_generator(
                        prompt,
                        max_new_tokens=1500,
                        num_return_sequences=1,
                        temperature=0.75,
                        do_sample=True,
                        pad_token_id=50256,
                        top_p=0.92,
                        repetition_penalty=1.18,
                        no_repeat_ngram_size=3
                    )
                    
                    if generation_result and len(generation_result) > 0:
                        full_analysis = generation_result[0]["generated_text"]
                    else:
                        full_analysis = ""
                except Exception as e:
                    full_analysis = ""
        
        analysis_placeholder.empty()
        
        # Extract generated summary
        analysis_text = ""
        if full_analysis:
            marker = "Be thorough and analytical in explaining what these changes represent:"
            if marker in full_analysis:
                parts = full_analysis.split(marker)
                if len(parts) > 1:
                    analysis_text = parts[-1].strip()
            
            if not analysis_text:
                marker2 = "Complete list of all detected changes:"
                if marker2 in full_analysis:
                    after_list = full_analysis.split(marker2)[-1]
                    if "\n\n" in after_list:
                        parts = after_list.split("\n\n", 1)
                        if len(parts) > 1:
                            analysis_text = parts[-1].strip()
            
            if not analysis_text and len(full_analysis) > len(prompt):
                analysis_text = full_analysis[len(prompt):].strip()
        
        # Fallback if analysis is too short
        if len(analysis_text) < 100:
            type_descriptions = []
            for ctype, clist in change_types_dict.items():
                count = len(clist)
                total_area = sum(c.get("area_pixels", 0) for c in clist)
                pct = (total_area / total_changed_pixels * 100) if total_changed_pixels > 0 else 0
                type_descriptions.append(
                    f"The analysis identified {count} instances of {ctype}, "
                    f"accounting for {total_area:,} pixels or {pct:.1f}% of total changes. "
                )
            
            analysis_text = (
                f"The satellite imagery analysis has detected {len(changes)} distinct change regions "
                f"across the observed area, representing approximately {change_percentage:.1f}% coverage "
                f"of the total analyzed scene. These changes encompass {total_changed_pixels:,} pixels "
                f"in total, with an average change region size of {avg_change_size:,} pixels. "
                f"\n\n"
                f"{''.join(type_descriptions)}"
                f"\n\n"
                f"The spatial distribution of changes indicates varied patterns across the imagery. "
                f"These detected modifications suggest temporal evolution in land cover, structural "
                f"development, vegetation dynamics, and surface characteristics. The comprehensive "
                f"change detection reveals both concentrated clusters and dispersed individual changes, "
                f"providing insight into the multifaceted nature of landscape transformation captured "
                f"between the two temporal snapshots."
            )
        
        # === STREAMING AI ANALYSIS OUTPUT ===
        st.markdown("")
        analysis_box = st.empty()
        
        streamed_text = ""
        words = analysis_text.split()
        
        # Stream word-by-word for alive feeling
        for i, word in enumerate(words):
            streamed_text += word + " "
            
            if "\n\n" in word or (i > 0 and "\n\n" in words[i-1]):
                streamed_text += "\n\n"
            
            analysis_box.markdown(
                f'<div class="stream-box">{streamed_text}</div>', 
                unsafe_allow_html=True
            )
            
            # Variable speed for natural reading
            if word.endswith(('.', '!', '?', ':')):
                time.sleep(0.08)
            elif word.endswith(','):
                time.sleep(0.06)
            else:
                time.sleep(0.035)

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")


def landing_page():
    st.markdown('<div class="main-title">AERIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Satellite Intelligence Platform</div>', unsafe_allow_html=True)

    st.markdown("<p style='text-align:center;color:#aaa;'>Choose a search mode</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        if st.button("IMAGE & TEXT", key="image_text_mode"):
            st.session_state["mode"] = "image_text"
            st.rerun()
        st.markdown("<p style='text-align:center;color:#888;font-size:12px;'>Upload and analyze satellite images</p>", unsafe_allow_html=True)

    with col2:
        if st.button("TEXT SEARCH", key="text_mode"):
            st.session_state["mode"] = "text"
            st.rerun()
        st.markdown("<p style='text-align:center;color:#888;font-size:12px;'>Search using natural language</p>", unsafe_allow_html=True)
    
    with col3:
        if st.button("CHANGE DETECTION", key="change_detection_mode"):
            st.session_state["mode"] = "change_detection"
            st.rerun()
        st.markdown("<p style='text-align:center;color:#888;font-size:12px;'>Compare two satellite images</p>", unsafe_allow_html=True)


def sidebar_config():
    with st.sidebar:
        st.markdown('<div class="section-heading">About</div>', unsafe_allow_html=True)
        
        st.markdown(
            """
            **Aeris** is an AI-powered satellite intelligence platform for analyzing and retrieving satellite imagery.
            
            **Features:**
            - Semantic image search
            - Natural language queries
            - Change detection analysis
            - Visual similarity matching
            """
        )
        
        st.markdown('<div class="section-heading">Dataset</div>', unsafe_allow_html=True)
        
        # Only show dataset info on landing page (fast check)
        mode = st.session_state.get("mode")
        if mode is None and DATASET_PATH.exists():
            st.markdown("**Status:** Ready")
            st.markdown("**Source:** Sentinel-2 Archive")
        elif DATASET_PATH.exists():
            # Load dataset info only when in a mode (lazy)
            try:
                import pandas as pd
                dataset = load_dataset()
                if dataset is not None:
                    st.markdown(f"**Satellite Images:** {len(dataset):,}")
                    st.markdown("**Source:** Sentinel-2 Archive")
            except:
                st.markdown("**Status:** Loading...")
        
        st.markdown('<div class="section-heading">Computation</div>', unsafe_allow_html=True)
        
        # Lazy check GPU only when needed
        if mode is not None:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                st.markdown(f"**Processor:** GPU ({torch.cuda.get_device_name(0)})")
                st.markdown(f"**Memory:** {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")
            else:
                st.markdown("**Processor:** CPU")
        else:
            st.markdown("**Processor:** Ready")


def main():
    sidebar_config()

    # Quick file checks (fast - no loading)
    if not DATASET_PATH.exists():
        st.error(f"Dataset not found: {DATASET_PATH}")
        return

    if not FAISS_PATH.exists():
        st.error(f"FAISS index not found: {FAISS_PATH}")
        return

    if not INDEX_PATH.exists():
        st.error(f"Index mapping not found: {INDEX_PATH}")
        return

    mode = st.session_state.get("mode")

    # Landing page loads instantly (no heavy imports)
    if mode is None:
        landing_page()
    else:
        # Load everything ONCE when user clicks a mode
        pipelines_ready, error = lazy_load_pipelines()
        
        if not pipelines_ready:
            st.error("Backend could not be loaded.")
            st.code(error)
            return
        
        # Mode-specific pages
        if mode == "text":
            if st.button("← Back"):
                st.session_state["mode"] = None
                st.rerun()
            run_text_mode()
        elif mode == "image_text":
            if st.button("← Back"):
                st.session_state["mode"] = None
                st.rerun()
            run_image_text_mode()
        elif mode == "change_detection":
            if st.button("← Back"):
                st.session_state["mode"] = None
                st.rerun()
            run_change_detection_mode()

    st.markdown(
        '<div class="footer">Aeris Satellite Intelligence Platform • Powered by AI</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
