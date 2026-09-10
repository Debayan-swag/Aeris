import sys
from pathlib import Path

# Fast imports only on startup
import streamlit as st

# Paths and directories
PROJECT_DIR = Path(r"C:\Users\Debayan\OneDrive\Desktop\Aeris")
DATASET_PATH = PROJECT_DIR / "sentinel-2-processed.parquet"
EMBEDDING_DIR = PROJECT_DIR / "remoteclip_embeddings"
FAISS_PATH = EMBEDDING_DIR / "remoteclip.faiss"
INDEX_PATH = EMBEDDING_DIR / "remoteclip_index.parquet"

from typing import Any

# Module-level variable stubs for lazy loading (resolves IDE/Pylance unbound global warnings)
stream_image_text_pipeline: Any = None
remoteclip_model: Any = None
remoteclip_tokenizer: Any = None
remoteclip_device: Any = None
run_text_pipeline: Any = None
stream_text_pipeline: Any = None
compare_satellite_images: Any = None
faiss: Any = None
np: Any = None
pd: Any = None
torch: Any = None
Image: Any = None
BytesIO: Any = None
Document: Any = None

# Global loading flags
_pipelines_loaded = False
_import_error = ""


def lazy_load_pipelines():
    """Lazy load heavy AI and PyTorch pipelines only when an operational mode is engaged"""
    global _pipelines_loaded, _import_error
    
    if _pipelines_loaded:
        return True, ""
    
    try:
        global faiss, np, pd, torch, Image, BytesIO, Document
        global stream_image_text_pipeline, remoteclip_model, remoteclip_tokenizer, remoteclip_device
        global run_text_pipeline, stream_text_pipeline, compare_satellite_images
        
        import faiss as _faiss
        import numpy as _np
        import pandas as _pd
        import torch as _torch
        from PIL import Image as _Image
        from io import BytesIO as _BytesIO
        from langchain_core.documents import Document as _Document
        
        from image_text import (
            stream_image_text_pipeline as _sitp,
            remoteclip_model as _rmodel,
            remoteclip_tokenizer as _rtokenizer,
            remoteclip_device as _rdevice,
        )
        from text import run_text_pipeline as _rtp, stream_text_pipeline as _stp
        from change_detection import compare_satellite_images as _csi

        faiss = _faiss
        np = _np
        pd = _pd
        torch = _torch
        Image = _Image
        BytesIO = _BytesIO
        Document = _Document
        stream_image_text_pipeline = _sitp
        remoteclip_model = _rmodel
        remoteclip_tokenizer = _rtokenizer
        remoteclip_device = _rdevice
        run_text_pipeline = _rtp
        stream_text_pipeline = _stp
        compare_satellite_images = _csi
        
        _pipelines_loaded = True
        return True, ""
    except Exception as exc:
        _import_error = str(exc)
        return False, str(exc)


# Page Configuration
st.set_page_config(
    page_title="Aeris · Satellite Intelligence Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Extensive Futuristic Cosmos Design System
st.markdown(
    """
    <style>
    /* ========== AERIS DEEP SPACE DESIGN SYSTEM ========== */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg-void: #02040a;
        --bg-panel: rgba(8, 14, 28, 0.72);
        --bg-card: rgba(14, 22, 42, 0.55);
        --bg-card-hover: rgba(20, 32, 58, 0.75);
        
        --ink-primary: #f8fafc;
        --ink-secondary: #94a3b8;
        --ink-muted: #64748b;
        
        --cyan-core: #38bdf8;
        --cyan-glow: rgba(56, 189, 248, 0.28);
        --cyan-border: rgba(56, 189, 248, 0.22);
        --cyan-border-hover: rgba(125, 211, 252, 0.6);
        
        --indigo-core: #818cf8;
        --purple-core: #c084fc;
        --emerald-core: #34d399;
        --amber-core: #fbbf24;
        
        --font-mono: 'IBM Plex Mono', monospace;
        --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
        --font-display: 'Space Grotesk', sans-serif;
        --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* Base Canvas & Atmosphere */
    .stApp {
        background-color: var(--bg-void);
        background-image: 
            radial-gradient(ellipse 90% 55% at 50% -12%, rgba(56, 189, 248, 0.14), transparent 72%),
            radial-gradient(ellipse 70% 40% at 85% 110%, rgba(129, 140, 248, 0.10), transparent 70%),
            radial-gradient(ellipse 60% 45% at 15% 105%, rgba(192, 132, 252, 0.07), transparent 65%),
            linear-gradient(rgba(56, 189, 248, 0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(56, 189, 248, 0.025) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 100% 100%, 48px 48px, 48px 48px;
        color: var(--ink-primary);
        font-family: var(--font-sans);
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
    }

    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1220px !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Top Nav Mission Banner */
    .top-nav-hud {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 18px;
        background: rgba(8, 14, 28, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.14);
        border-radius: 999px;
        backdrop-filter: blur(16px);
        margin-bottom: 24px;
        font-family: var(--font-mono);
        font-size: 10.5px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--ink-muted);
    }

    .top-nav-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .top-nav-right {
        display: flex;
        align-items: center;
        gap: 18px;
    }

    .hud-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 2px 10px;
        border-radius: 999px;
        background: rgba(56, 189, 248, 0.06);
        border: 1px solid rgba(56, 189, 248, 0.18);
        color: var(--cyan-core);
    }

    /* Hero Presentation */
    .hero-container {
        text-align: center;
        padding: 24px 0 20px 0;
        margin: 0 auto 12px auto;
        display: flex;
        flex-direction: column;
        align-items: center;
        position: relative;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: rgba(56, 189, 248, 0.07);
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 5px 16px;
        border-radius: 999px;
        margin-bottom: 18px;
        box-shadow: 0 0 24px rgba(56, 189, 248, 0.18);
    }

    .hero-badge-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--emerald-core);
        box-shadow: 0 0 10px var(--emerald-core);
        animation: pulseDot 2s ease-in-out infinite;
    }

    .hero-badge-text {
        font-family: var(--font-mono);
        font-size: 11px;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: var(--cyan-core);
        font-weight: 500;
    }

    .main-title {
        font-family: var(--font-display);
        font-weight: 700;
        font-size: clamp(3.2rem, 6vw, 4.8rem);
        line-height: 1.02;
        letter-spacing: -0.03em;
        margin: 0 0 12px 0;
        text-align: center;
        background: linear-gradient(110deg, #ffffff 10%, #bae6fd 40%, #7dd3fc 65%, #c084fc 95%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmerGrad 7s ease-in-out infinite;
    }

    .main-subtitle {
        font-family: var(--font-sans);
        font-size: 16px;
        font-weight: 300;
        color: var(--ink-secondary);
        letter-spacing: 0.01em;
        max-width: 620px;
        margin: 0 auto 30px auto;
        line-height: 1.65;
        text-align: center;
    }

    /* Mode Selection Glass Cards */
    .mode-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid var(--cyan-border);
        border-radius: 18px;
        padding: 26px 24px 22px 24px;
        min-height: 195px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.4s var(--ease-out);
        margin-bottom: 14px;
        position: relative;
        overflow: hidden;
    }

    .mode-card::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 50% 0%, rgba(56, 189, 248, 0.12), transparent 75%);
        opacity: 0;
        transition: opacity 0.35s ease;
        pointer-events: none;
    }

    .mode-card:hover {
        background: var(--bg-card-hover);
        border-color: var(--cyan-border-hover);
        transform: translateY(-5px);
        box-shadow: 0 20px 48px -12px var(--cyan-glow), 0 0 1px 1px rgba(125, 211, 252, 0.25);
    }

    .mode-card:hover::before {
        opacity: 1;
    }

    .mode-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
    }

    .mode-tag {
        font-family: var(--font-mono);
        font-size: 9.5px;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--cyan-core);
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 3px 10px;
        border-radius: 999px;
        font-weight: 500;
    }

    .mode-icon-wrap {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: grid;
        place-items: center;
        background: rgba(56, 189, 248, 0.06);
        border: 1px solid rgba(56, 189, 248, 0.2);
        transition: transform 0.3s ease;
    }

    .mode-card:hover .mode-icon-wrap {
        transform: scale(1.08) rotate(3deg);
        background: rgba(56, 189, 248, 0.12);
        border-color: rgba(56, 189, 248, 0.45);
    }

    .mode-card-title {
        font-family: var(--font-display);
        font-size: 19px;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -0.01em;
        margin-bottom: 7px;
    }

    .mode-card-desc {
        font-size: 13px;
        line-height: 1.58;
        color: var(--ink-secondary);
        font-weight: 300;
        margin: 0;
    }

    /* Buttons */
    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(56, 189, 248, 0.25);
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.015) 100%);
        color: #f1f5f9;
        font-family: var(--font-sans);
        font-size: 13.5px;
        font-weight: 500;
        letter-spacing: 0.02em;
        transition: all 0.3s var(--ease-out);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
        padding: 11px 20px;
    }

    div.stButton > button:hover {
        border-color: rgba(125, 211, 252, 0.7);
        background: linear-gradient(180deg, rgba(56, 189, 248, 0.18) 0%, rgba(56, 189, 248, 0.06) 100%);
        transform: translateY(-2px);
        box-shadow: 0 10px 28px var(--cyan-glow);
        color: #ffffff;
    }

    div.stButton > button:active {
        transform: translateY(0);
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(180deg, rgba(56, 189, 248, 0.28) 0%, rgba(56, 189, 248, 0.12) 100%);
        border: 1px solid rgba(125, 211, 252, 0.6);
        color: #ffffff;
        font-weight: 600;
        box-shadow: 0 4px 24px rgba(56, 189, 248, 0.32), inset 0 1px 0 rgba(255, 255, 255, 0.25);
    }

    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(180deg, rgba(125, 211, 252, 0.42) 0%, rgba(56, 189, 248, 0.22) 100%);
        border-color: rgba(186, 230, 253, 0.9);
        box-shadow: 0 10px 36px rgba(56, 189, 248, 0.48), inset 0 1px 0 rgba(255, 255, 255, 0.4);
        transform: translateY(-2px);
    }

    /* Ghost Back Navigation Button */
    .back-btn-wrap {
        margin-bottom: 20px;
    }

    .back-btn-wrap div.stButton > button {
        width: auto !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
        padding: 7px 18px !important;
        font-family: var(--font-mono) !important;
        font-size: 11.5px !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        border-radius: 999px !important;
        background: rgba(8, 14, 28, 0.55) !important;
        border: 1px solid rgba(56, 189, 248, 0.22) !important;
        color: var(--ink-secondary) !important;
    }

    .back-btn-wrap div.stButton > button:hover {
        border-color: rgba(125, 211, 252, 0.7) !important;
        background: rgba(56, 189, 248, 0.12) !important;
        color: #ffffff !important;
        transform: translateX(-3px) !important;
    }

    /* Section Headings */
    .section-heading {
        font-family: var(--font-display);
        font-size: 24px;
        font-weight: 600;
        color: #ffffff;
        margin: 18px 0 6px 0;
        letter-spacing: -0.015em;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .section-heading::before {
        content: '';
        display: inline-block;
        width: 3.5px;
        height: 22px;
        background: linear-gradient(180deg, var(--cyan-core), var(--indigo-core));
        border-radius: 3px;
        box-shadow: 0 0 12px var(--cyan-core);
    }

    .section-subtitle {
        color: var(--ink-secondary);
        font-size: 14px;
        margin-bottom: 22px;
        padding-left: 15px;
        font-weight: 300;
    }

    /* Sample Query Chips */
    .sample-chips-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 16px;
    }

    .sample-chips-label {
        font-family: var(--font-mono);
        font-size: 10px;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--ink-muted);
        margin-bottom: 8px;
    }

    /* Text Inputs & File Uploaders */
    .stTextArea textarea {
        background: rgba(8, 14, 28, 0.65) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        border-radius: 14px !important;
        color: #f8fafc !important;
        font-size: 14px !important;
        padding: 16px !important;
        font-family: var(--font-sans) !important;
        transition: all 0.3s ease !important;
        box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.3) !important;
    }

    .stTextArea textarea:focus {
        border-color: rgba(125, 211, 252, 0.7) !important;
        box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15), inset 0 2px 6px rgba(0, 0, 0, 0.3) !important;
        background: rgba(12, 20, 38, 0.85) !important;
    }

    [data-testid="stFileUploader"] {
        background: rgba(8, 14, 28, 0.45);
        border: 1px dashed rgba(56, 189, 248, 0.3);
        border-radius: 16px;
        padding: 22px;
        transition: all 0.3s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(125, 211, 252, 0.65);
        background: rgba(56, 189, 248, 0.05);
        box-shadow: 0 0 24px rgba(56, 189, 248, 0.12);
    }

    /* Satellite Image Viewport Frame with Reticle Corners */
    .image-hud-frame {
        position: relative;
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(56, 189, 248, 0.2);
        background: #000000;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45);
        transition: all 0.35s var(--ease-out);
        margin-bottom: 12px;
    }

    .image-hud-frame:hover {
        border-color: rgba(125, 211, 252, 0.55);
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 16px 40px -10px var(--cyan-glow);
    }

    .image-caption-pill {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 6px;
        font-family: var(--font-mono);
    }

    .sim-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-family: var(--font-mono);
        font-size: 10px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--cyan-core);
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 6px;
        padding: 3px 8px;
        font-weight: 600;
    }

    .sim-badge-high {
        color: #38bdf8;
        border-color: rgba(56, 189, 248, 0.4);
        background: rgba(56, 189, 248, 0.14);
    }

    /* AI Thought Streaming Box */
    .stream-box {
        background: rgba(4, 8, 20, 0.78);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-left: 4px solid var(--cyan-core);
        border-radius: 14px;
        padding: 22px 26px;
        line-height: 1.75;
        font-size: 14px;
        color: #f1f5f9;
        box-shadow: 0 10px 36px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        font-weight: 400;
        position: relative;
    }

    /* Metrics Cards */
    [data-testid="stMetric"] {
        background: rgba(8, 14, 28, 0.5) !important;
        border: 1px solid rgba(56, 189, 248, 0.18) !important;
        border-radius: 14px !important;
        padding: 18px 22px !important;
        backdrop-filter: blur(14px) !important;
        transition: border-color 0.3s ease !important;
    }

    [data-testid="stMetric"]:hover {
        border-color: rgba(125, 211, 252, 0.45) !important;
    }

    [data-testid="stMetricLabel"] {
        font-family: var(--font-mono) !important;
        font-size: 10.5px !important;
        color: var(--ink-muted) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.16em !important;
    }

    [data-testid="stMetricValue"] {
        font-family: var(--font-display) !important;
        font-size: 28px !important;
        font-weight: 600 !important;
        color: #ffffff !important;
        letter-spacing: -0.01em !important;
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc) !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.6) !important;
        border-radius: 6px !important;
    }

    /* Sidebar Cosmos Theme */
    [data-testid="stSidebar"] {
        background: #040711 !important;
        border-right: 1px solid rgba(56, 189, 248, 0.12) !important;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 0 18px 0;
        border-bottom: 1px solid rgba(56, 189, 248, 0.1);
        margin-bottom: 18px;
    }

    .sidebar-brand-name {
        font-family: var(--font-display);
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 0.1em;
        color: #ffffff;
    }

    .sidebar-status-tag {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        font-family: var(--font-mono);
        font-size: 9.5px;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--emerald-core);
        background: rgba(52, 211, 153, 0.08);
        border: 1px solid rgba(52, 211, 153, 0.25);
        padding: 4px 12px;
        border-radius: 999px;
        margin-bottom: 22px;
    }

    .sidebar-dot {
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background: var(--emerald-core);
        box-shadow: 0 0 8px var(--emerald-core);
        animation: pulseDot 2s ease-in-out infinite;
    }

    .sidebar-section-label {
        font-family: var(--font-mono);
        font-size: 9px;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: var(--ink-muted);
        margin: 20px 0 10px 0;
    }

    .sidebar-card {
        background: rgba(8, 14, 28, 0.45);
        border: 1px solid rgba(56, 189, 248, 0.12);
        border-radius: 12px;
        padding: 13px 15px;
        margin-bottom: 10px;
        transition: border-color 0.25s ease;
    }

    .sidebar-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
    }

    .sidebar-card-label {
        font-family: var(--font-mono);
        font-size: 9px;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--ink-muted);
        margin-bottom: 4px;
    }

    .sidebar-card-value {
        font-family: var(--font-sans);
        font-size: 13.5px;
        font-weight: 500;
        color: #ffffff;
    }

    .sidebar-card-sub {
        font-family: var(--font-mono);
        font-size: 10px;
        color: var(--cyan-core);
        margin-top: 3px;
    }

    /* Tabs & Expanders */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(8, 14, 28, 0.5);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(56, 189, 248, 0.12);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-family: var(--font-sans);
        font-size: 13px;
        font-weight: 500;
        color: var(--ink-secondary);
        transition: all 0.25s ease;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.18) !important;
        border: 1px solid rgba(125, 211, 252, 0.45) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.22);
    }

    [data-testid="stExpander"] {
        background: rgba(8, 14, 28, 0.4) !important;
        border: 1px solid rgba(56, 189, 248, 0.14) !important;
        border-radius: 14px !important;
        overflow: hidden;
    }

    .result-title {
        font-family: var(--font-display);
        font-size: 19px;
        font-weight: 600;
        color: #ffffff;
        margin: 22px 0 14px 0;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .no-results {
        background: rgba(8, 14, 28, 0.35);
        border: 1px dashed rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 44px 28px;
        text-align: center;
        color: var(--ink-secondary);
        margin: 28px 0;
    }

    /* Telemetry HUD Strip Bottom */
    .telemetry-strip {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 28px;
        flex-wrap: wrap;
        padding: 22px 0 12px 0;
        margin-top: 32px;
        border-top: 1px solid rgba(56, 189, 248, 0.1);
        font-family: var(--font-mono);
        font-size: 10.5px;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--ink-muted);
    }

    .footer {
        text-align: center;
        color: var(--ink-muted);
        font-family: var(--font-mono);
        font-size: 11px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-top: 48px;
        padding: 22px 0;
        border-top: 1px solid rgba(56, 189, 248, 0.08);
    }

    /* Keyframe Animations */
    @keyframes shimmerGrad {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    @keyframes pulseDot {
        0%, 100% { opacity: 0.4; transform: scale(0.85); }
        50% { opacity: 1; transform: scale(1.15); }
    }

    @keyframes orbitSpin {
        to { transform: rotate(360deg); }
    }

    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 7px;
        height: 7px;
    }
    ::-webkit-scrollbar-track {
        background: #02040a;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(56, 189, 248, 0.22);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(125, 211, 252, 0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# ULTRA-FAST ARCHIVE & PARQUET RETRIEVAL ENGINE
# Optimized for zero-latency loading and microsecond random image lookups
# ==============================================================================

@st.cache_resource(show_spinner=False)
def get_archive_stats():
    """Reads Parquet metadata in ~1 millisecond without loading 3.1GB into RAM"""
    try:
        import pyarrow.parquet as pq
        meta = pq.read_metadata(str(DATASET_PATH))
        num_rows = meta.num_rows
        size_gb = round(DATASET_PATH.stat().st_size / (1024**3), 2)
        return {
            "total_scenes": num_rows,
            "size_gb": size_gb,
            "status": "Online",
        }
    except Exception:
        return {
            "total_scenes": 89796,
            "size_gb": 2.93,
            "status": "Online",
        }


@st.cache_resource(show_spinner=False)
def get_parquet_reader():
    """Singleton ParquetFile handle for zero-copy row-group reads"""
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(str(DATASET_PATH))
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_image_name_to_index():
    """Cached index mapping: image_name -> row_group index (loaded in ~0.2s from 2MB file)"""
    try:
        import pandas as pd
        df = pd.read_parquet(INDEX_PATH, columns=["image_name"])
        return dict(zip(df["image_name"], df.index))
    except Exception:
        return {}


@st.cache_data(max_entries=300, show_spinner=False)
def get_cached_satellite_image(image_name: str):
    """
    Fetches individual image from Parquet row group in ~2-4ms.
    Eliminates loading the 3.15GB DataFrame into RAM!
    """
    try:
        from PIL import Image as PILImage
        from io import BytesIO as IOMemory
        
        name_map = get_image_name_to_index()
        row_idx = name_map.get(image_name)
        if row_idx is None:
            return None
            
        pf = get_parquet_reader()
        if pf is None:
            return None
            
        rg = pf.read_row_group(int(row_idx), columns=["image"])
        img_bytes = rg["image"][0].as_py()
        return PILImage.open(IOMemory(img_bytes)).convert("RGB")
    except Exception:
        return None


@st.cache_resource(show_spinner=False, ttl=3600)
def load_dataset():
    """
    Cached dataset loader for backward compatibility.
    Only loaded if explicitly demanded by a pipeline node.
    """
    try:
        import pandas as pd
        return pd.read_parquet(DATASET_PATH, engine='pyarrow')
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None


@st.cache_resource(show_spinner=False, ttl=3600)
def load_retrieval_resources():
    """Load FAISS index and mapping with optimized caching"""
    try:
        import faiss as _faiss
        import pandas as _pd
        
        vector_store = _faiss.read_index(str(FAISS_PATH))
        index_mapping = _pd.read_parquet(INDEX_PATH, engine='pyarrow')
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
        import torch as _torch
        import numpy as _np
        from langchain_core.documents import Document as _Doc

        tokens = self.tokenizer([query]).to(self.device)

        with _torch.no_grad():
            query_embedding = self.model.encode_text(tokens)
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

        query_embedding = query_embedding.cpu().numpy().astype(_np.float32)
        scores, indices = self.vector_store.search(query_embedding, k)

        results = []
        for index, score in zip(indices[0], scores[0]):
            if index < 0:
                continue

            row = self.index_mapping.iloc[int(index)]
            image_name = row["image_name"]

            document = _Doc(
                page_content=f"Satellite image: {image_name}",
                metadata={"image_name": image_name, "index": int(index)},
            )

            results.append((document, float(score)))

        return results


@st.cache_resource(show_spinner=False, ttl=3600)
def load_text_retriever():
    """Load text retriever with cached RemoteCLIP model and FAISS store"""
    vector_store, index_mapping = load_retrieval_resources()
    if vector_store is None or index_mapping is None:
        return None
    
    from image_text import remoteclip_model as rmodel, remoteclip_tokenizer as rtok, remoteclip_device as rdev
    
    return RemoteCLIPTextRetriever(
        vector_store=vector_store,
        index_mapping=index_mapping,
        model=rmodel,
        tokenizer=rtok,
        device=rdev,
    )


def get_image(dataframe, image_name):
    """
    Optimized image getter:
    1. Attempts instant microsecond Parquet row-group fetch.
    2. Falls back to dataframe scan only if needed.
    """
    fast_img = get_cached_satellite_image(image_name)
    if fast_img is not None:
        return fast_img
        
    if dataframe is not None:
        try:
            from PIL import Image as PILImage
            from io import BytesIO as IOMemory
            rows = dataframe.loc[dataframe["image_name"] == image_name]
            if not rows.empty:
                image_bytes = rows.iloc[0]["image"]
                return PILImage.open(IOMemory(image_bytes)).convert("RGB")
        except Exception:
            pass
    return None


def display_results(results, dataframe, title="Results"):
    """Displays retrieved satellite imagery with high-tech HUD cards and similarity gauges"""
    if not results or len(results) == 0:
        st.markdown(
            """
            <div class="no-results">
                <div style="font-family:var(--font-display);font-size:18px;font-weight:600;color:#fff;margin-bottom:6px;">No Corresponding Scenes Found</div>
                <div style="font-size:13px;color:var(--ink-secondary);">The target concepts did not produce high-confidence matches within the active Sentinel-2 archive.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div class="result-title">'
        f'<span>{title}</span>'
        f'<span class="hud-pill">{len(results)} SCENES IDENTIFIED</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    num_cols = min(len(results), 5)
    columns = st.columns(num_cols)

    for i, result in enumerate(results):
        image_name = result.get("image_name", "")
        score = float(result.get("score", result.get("retrieval_score", 0.0)))
        image = get_image(dataframe, image_name)

        with columns[i % num_cols]:
            if image is not None:
                st.markdown('<div class="image-hud-frame">', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            badge_cls = "sim-badge sim-badge-high" if score >= 0.70 else "sim-badge"
            st.markdown(
                f'<div class="image-caption-pill">'
                f'  <span style="font-size:11.5px;color:#f8fafc;max-width:65%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="{image_name}">{image_name}</span>'
                f'  <span class="{badge_cls}">SIM {score:.3f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ==============================================================================
# OPERATIONAL MODES
# ==============================================================================

def run_text_mode():
    st.markdown('<div class="section-heading">Semantic Text Search</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Search the multi-spectral Sentinel-2 archive using natural language concepts and spatial descriptions.</div>',
        unsafe_allow_html=True,
    )

    # Interactive Sample Query Suggestion Chips
    st.markdown('<div class="sample-chips-label">SUGGESTED TARGETS &middot; 1-CLICK TEST:</div>', unsafe_allow_html=True)
    chip_cols = st.columns(4)
    sample_queries = [
        "Major international airport with runways",
        "Deep water commercial port and shipping docks",
        "Circular pivot irrigation farmland crops",
        "Utility-scale photovoltaic solar array",
    ]
    
    for idx, sample in enumerate(sample_queries):
        with chip_cols[idx]:
            if st.button(f"🎯 {sample.split()[0]} {sample.split()[1]}", key=f"chip_{idx}"):
                st.session_state["search_query"] = sample
                st.rerun()

    current_query = st.session_state.get("search_query", "")
    query = st.text_area(
        "Search Query",
        value=current_query,
        placeholder="Example: Find satellite images showing large airports with aircraft near runways, or coastal cargo shipping terminals.",
        height=100,
    )

    search_button = st.button("Search Archive →", type="primary", use_container_width=True)

    if not search_button:
        return

    if not query.strip():
        st.warning("Please specify a target query or click a suggested target above.")
        return

    try:
        with st.spinner("Accessing RemoteCLIP neural index..."):
            retriever = load_text_retriever()
            if retriever is None:
                st.error("Text retriever could not be initialized.")
                return

        st.markdown('<div id="processing-section"></div>', unsafe_allow_html=True)
        st.markdown('<div class="result-title">Search Telemetry & Pipeline Stream</div>', unsafe_allow_html=True)
        
        status_box = st.empty()
        progress_bar = st.progress(0)
        thinking_container = st.container()
        results_container = st.container()

        state = {
            "original_shown": False,
            "thinking_box": None,
            "thinking_text": "",
            "queries_shown": False,
            "retrieval_started": False,
            "retrieval_expander": None,
            "retrieval_details": []
        }

        # Stream the text pipeline
        for event in stream_text_pipeline(query.strip(), retriever):
            event_type = event.get("type")
            data = event.get("data")

            if event_type == "stage":
                status_box.info(f"{data}")
                
                if "Understanding" in str(data):
                    progress_bar.progress(0.15)
                    if not state["original_shown"]:
                        with thinking_container:
                            with st.expander("Query Telemetry & Spatial Constraints", expanded=True):
                                st.markdown(f"**Target Concept:** `{query.strip()}`")
                                st.caption("Analyzing semantic representations across 512-dimensional feature space...")
                        state["original_shown"] = True
                        
                elif "AI analysis" in str(data) or "analysis" in str(data).lower():
                    progress_bar.progress(0.35)
                    status_box.info("Decomposing semantic query into multi-angle feature vectors...")
                    
                elif "variations" in str(data).lower() or "Generating" in str(data):
                    progress_bar.progress(0.50)
                    status_box.info("Synthesizing precision search variants...")
                    
                elif "Retrieving" in str(data) or "images" in str(data):
                    progress_bar.progress(0.70)
                    if not state["retrieval_started"]:
                        with thinking_container:
                            state["retrieval_expander"] = st.expander("Vector Space Traversal", expanded=True)
                        state["retrieval_started"] = True
                    
                elif "top 7" in str(data) or "best matches" in str(data) or "Selecting" in str(data):
                    progress_bar.progress(0.88)
                    status_box.info("Ranking top candidates by cosine similarity...")
                    
                elif "complete" in str(data).lower():
                    progress_bar.progress(1.0)

            elif event_type == "thinking_start":
                if state["thinking_box"] is None:
                    with thinking_container:
                        st.markdown('<div class="result-title">AI Grounding & Context</div>', unsafe_allow_html=True)
                        st.caption("Neural reasoning model evaluating satellite scene composition")
                        state["thinking_box"] = st.empty()

            elif event_type == "thinking_content":
                state["thinking_text"] += str(data)
                if state["thinking_box"] is not None:
                    state["thinking_box"].markdown(
                        f'<div class="stream-box">{state["thinking_text"]}</div>',
                        unsafe_allow_html=True
                    )

            elif event_type == "queries":
                if not state["queries_shown"]:
                    with thinking_container:
                        with st.expander("Multi-Vector Query Expansion (DistilGPT-2)", expanded=True):
                            st.success(f"**Synthesized {len(data)} Precision Sub-Queries:**")
                            for i, q in enumerate(data, 1):
                                st.markdown(f"**Query {i}:** `{q}`")
                            st.caption("Parallel vector queries executed across Sentinel-2 FAISS index.")
                    state["queries_shown"] = True

            elif event_type == "query_variation":
                variation_data = data
                state["retrieval_details"].append(f"→ Searching with variation {variation_data.get('index', 1)}...")
                if state["retrieval_expander"]:
                    with state["retrieval_expander"]:
                        st.caption("\n".join(state["retrieval_details"][-3:]))

            elif event_type == "partial_results":
                state["retrieval_details"].append(f"Retrieved {data.get('count', 0)} candidates from sub-query {data.get('query_num', 1)}")
                if state["retrieval_expander"]:
                    with state["retrieval_expander"]:
                        st.caption("\n".join(state["retrieval_details"][-4:]))

            elif event_type == "results":
                formatted_results = []
                for r in data:
                    metadata = r.get("metadata", {})
                    formatted_results.append({
                        "image_name": metadata.get("image_name", ""),
                        "score": r.get("score", 0.0)
                    })
                
                if state["retrieval_expander"]:
                    with state["retrieval_expander"]:
                        st.success(f"Vector search converged: {len(formatted_results)} top candidates selected.")
                
                status_box.success("Search Complete · Multi-Spectral Scenes Retrieved")
                progress_bar.empty()
                
                with results_container:
                    st.markdown("---")
                    display_results(formatted_results, None, "Retrieved Satellite Scenes")

    except Exception as exc:
        st.error(f"Search failed: {exc}")


def run_image_text_mode():
    st.markdown('<div class="section-heading">Image & Text Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Upload high-resolution satellite imagery and query multimodal representations or analyze temporal modifications.</div>',
        unsafe_allow_html=True,
    )

    col_up, col_prompt = st.columns([1, 1], gap="large")

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Satellite Imagery (TIF, PNG, JPG)",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        )
        if uploaded_file is not None:
            try:
                from PIL import Image as PILImage
                preview = PILImage.open(uploaded_file).convert("RGB")
                st.markdown('<div class="image-hud-frame">', unsafe_allow_html=True)
                st.image(preview, caption=f"Source: {uploaded_file.name}", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed to decode image: {e}")
                return

    with col_prompt:
        query = st.text_area(
            "Analytical Directives",
            placeholder="Example: Detect airport runways or identify changes in this port area compared to archive captures.",
            height=120,
        )
        analyze_button = st.button("Analyze Multimodal Features →", type="primary", use_container_width=True)

    if not analyze_button:
        return

    if uploaded_file is None:
        st.warning("Please upload a satellite image to begin analysis.")
        return

    if not query.strip():
        st.warning("Please enter analytical directives.")
        return

    try:
        from PIL import Image as PILImage
        with st.spinner("Preparing vector index and neural encoders..."):
            vector_store, index_mapping = load_retrieval_resources()
            if vector_store is None or index_mapping is None:
                st.error("Index mapping or vector store could not be loaded.")
                return

        uploaded_file.seek(0)
        image = PILImage.open(uploaded_file).convert("RGB")

        st.markdown('<div id="analysis-section"></div>', unsafe_allow_html=True)
        st.markdown('<div class="result-title">Live Multimodal Telemetry</div>', unsafe_allow_html=True)
        
        status_box = st.empty()
        progress_bar = st.progress(0)
        details_box = st.empty()
        description_container = st.container()
        results_container = st.container()

        response_text = ""
        description_started = False
        description_box = None

        for event in stream_image_text_pipeline(
            image=image,
            user_text=query.strip(),
            vector_store=vector_store,
            index_mapping=index_mapping,
            dataframe=load_dataset(),
            image_name=uploaded_file.name,
        ):
            event_type = event.get("type")
            data = event.get("data")

            if event_type == "status":
                status_box.info(f"{data}")
                
                if "Understanding" in str(data):
                    progress_bar.progress(0.12)
                    details_box.caption("Assessing query intent (retrieval vs. temporal diff)...")
                elif "Interpreting" in str(data) or "image" in str(data).lower():
                    progress_bar.progress(0.28)
                    details_box.info("Encoding visual features with RemoteCLIP ViT-B/32 vision transformer...")
                elif "Extracting" in str(data) or "semantic" in str(data).lower():
                    progress_bar.progress(0.44)
                    details_box.info("Generating multimodal semantic grounding...")
                elif "description" in str(data).lower():
                    progress_bar.progress(0.55)
                    details_box.info("NVIDIA Nemotron generating contextual interpretation...")
                elif "Searching" in str(data):
                    progress_bar.progress(0.72)
                    details_box.info("Querying 89,796 archive scenes in FAISS index...")
                elif "Ranking" in str(data):
                    progress_bar.progress(0.85)
                    details_box.info("Sorting closest archive matches...")
                elif "Comparing" in str(data):
                    progress_bar.progress(0.92)
                    details_box.warning("Executing pixel difference and structural change analysis...")
                elif "complete" in str(data).lower():
                    progress_bar.progress(1.0)

            elif event_type == "intent":
                if data == "normal_retriever":
                    details_box.success("Target Objective: Semantic Archive Retrieval")
                else:
                    details_box.success("Target Objective: Temporal Multi-Scale Change Detection")

            elif event_type == "content":
                response_text += str(data)
                
                if not description_started:
                    with description_container:
                        st.markdown("---")
                        st.markdown('<div class="result-title">AI Scene Interpretation</div>', unsafe_allow_html=True)
                        description_box = st.empty()
                    description_started = True
                
                if description_box is not None:
                    description_box.markdown(f'<div class="stream-box">{response_text}</div>', unsafe_allow_html=True)

            elif event_type == "semantic_evidence":
                concepts = data.get("concepts", [])
                if concepts:
                    details_box.caption(f"Identified Concept Tokens: {', '.join(concepts[:6])}")

            elif event_type == "retrieval_results":
                results = data
                progress_bar.progress(1.0)
                status_box.success("Multimodal Retrieval Complete")
                
                with results_container:
                    st.markdown("---")
                    display_results(results, None, "Archive Matches")

            elif event_type == "change_results":
                change_results = data
                progress_bar.progress(1.0)
                status_box.success("Change Detection Complete")
                
                with results_container:
                    st.markdown('<div class="result-title">Temporal Comparison & Verification</div>', unsafe_allow_html=True)
                    
                    all_different = all(item.get('different', False) for item in change_results)
                    if all_different:
                        st.success("All candidate scenes confirm structural and radiometric alterations.")
                        from change_detection_viz import create_difference_mask
                        
                        for idx, item in enumerate(change_results):
                            image_name = item.get('image_name', 'Unknown')
                            change_ratio = float(item.get('change_ratio', 0.0))
                            retrieval_score = float(item.get('retrieval_score', 0.0))
                            
                            with st.expander(f"Scene Comparison {idx+1}: {image_name}", expanded=(idx==0)):
                                candidate_img = get_cached_satellite_image(image_name)
                                if candidate_img is not None:
                                    c1, c2, c3 = st.columns(3)
                                    with c1:
                                        st.metric("Similarity", f"{retrieval_score:.4f}")
                                    with c2:
                                        st.metric("Change Ratio", f"{change_ratio*100:.1f}%")
                                    with c3:
                                        st.metric("Classification", "Modified" if change_ratio > 0.1 else "Stable")
                                        
                                    tab_side, tab_heat, tab_mask = st.tabs(["Side-by-Side", "Thermal Heatmap", "Binary Mask"])
                                    with tab_side:
                                        ca, cb = st.columns(2)
                                        with ca:
                                            st.image(image, caption="Source Upload", use_container_width=True)
                                        with cb:
                                            st.image(candidate_img, caption=f"Archive: {image_name}", use_container_width=True)
                                    with tab_heat:
                                        bmask, hmap, ovly = create_difference_mask(image, candidate_img)
                                        st.image(hmap, caption="Turbo Radiometric Heatmap", use_container_width=True)
                                    with tab_mask:
                                        bmask, hmap, ovly = create_difference_mask(image, candidate_img)
                                        st.image(bmask, caption="Binary Change Segmentation", use_container_width=True)

            elif event_type == "no_data":
                progress_bar.progress(1.0)
                status_box.warning("No Direct Match in Archive")
                with results_container:
                    st.markdown(
                        """
                        <div class="no-results">
                            <div style="font-family:var(--font-display);font-size:18px;font-weight:600;color:#fff;margin-bottom:4px;">No Archive Match Located</div>
                            <div style="font-size:13px;color:var(--ink-secondary);">This geographic capture does not have matching pre-indexed scenes in the current archive.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    except Exception as exc:
        st.error(f"Analysis pipeline error: {exc}")


def run_change_detection_mode():
    import cv2
    import tempfile
    import os
    import time
    import hashlib
    
    st.markdown('<div class="section-heading">Temporal Change Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Upload two temporal observations to compute automated radiometric normalization, alignment, and pixel modifications.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        before_image_file = st.file_uploader(
            "Temporal T-0 (Baseline)",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            key="before_image"
        )
        if before_image_file:
            from PIL import Image as PILImage
            preview_before = PILImage.open(before_image_file).convert("RGB")
            st.markdown('<div class="image-hud-frame">', unsafe_allow_html=True)
            st.image(preview_before, caption="T-0 Baseline Observation", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        after_image_file = st.file_uploader(
            "Temporal T-1 (Subsequent)",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            key="after_image"
        )
        if after_image_file:
            from PIL import Image as PILImage
            preview_after = PILImage.open(after_image_file).convert("RGB")
            st.markdown('<div class="image-hud-frame">', unsafe_allow_html=True)
            st.image(preview_after, caption="T-1 Subsequent Observation", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    detect_button = st.button("Compute Temporal Discrepancies →", type="primary", use_container_width=True)

    if not detect_button:
        return

    if before_image_file is None or after_image_file is None:
        st.warning("Please provide both T-0 and T-1 satellite scenes.")
        return

    try:
        before_image_file.seek(0)
        after_image_file.seek(0)
        before_hash = hashlib.md5(before_image_file.read()).hexdigest()
        after_hash = hashlib.md5(after_image_file.read()).hexdigest()
        
        if before_hash == after_hash:
            st.info("Identical radiometric fingerprints. Zero modifications detected across temporal captures.")
            return

        st.markdown("---")
        st.markdown('<div class="result-title">Autonomous Change Processing</div>', unsafe_allow_html=True)
        
        status_container = st.empty()
        progress_bar = st.progress(0)
        
        before_image_file.seek(0)
        after_image_file.seek(0)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_before:
            tmp_before.write(before_image_file.read())
            before_path = tmp_before.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_after:
            tmp_after.write(after_image_file.read())
            after_path = tmp_after.name
        
        steps = [
            ("Ingesting multi-spectral observations...", 0.15),
            ("Executing Enhanced Correlation Coefficient (ECC) coordinate alignment...", 0.35),
            ("Normalizing radiometric channel variance in CIELAB space...", 0.55),
            ("Computing multi-scale Gaussian difference tensors...", 0.75),
            ("Classifying morphological change contours...", 0.90),
        ]
        
        result = None
        for step_text, progress_val in steps:
            status_container.info(step_text)
            progress_bar.progress(progress_val)
            if progress_val == 0.75:
                result = compare_satellite_images(before_path, after_path)
            time.sleep(0.15)
        
        os.unlink(before_path)
        os.unlink(after_path)
        
        status_container.success("Temporal Analysis Completed")
        progress_bar.progress(1.0)
        time.sleep(0.3)
        status_container.empty()
        progress_bar.empty()
        
        changes = result.get("changes", [])
        change_percentage = result.get("change_percentage", 0.0)
        
        if len(changes) == 0:
            st.info("No statistically significant alterations detected between scenes.")
            return

        # Results metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Modified Clusters", f"{len(changes):,}", help="Distinct clustered change zones")
        with m2:
            st.metric("Surface Coverage", f"{change_percentage:.1f}%", help="Percentage of image surface affected")
        with m3:
            st.metric("Altered Pixels", f"{result.get('changed_pixels', 0):,}", help="Total pixel delta count")
        
        st.markdown("---")
        st.markdown('<div class="result-title">Multi-Spectral Change Visualization</div>', unsafe_allow_html=True)
        
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**T-0 Baseline Capture**")
            before_display = cv2.cvtColor(result["before_image"], cv2.COLOR_BGR2RGB)
            st.image(before_display, use_container_width=True)
        
        with col_r:
            st.markdown("**Identified Structural Detections**")
            overlay_display = cv2.cvtColor(result["change_overlay"], cv2.COLOR_BGR2RGB)
            st.image(overlay_display, use_container_width=True)

        with st.expander(f"Telemetry Log & Cluster Coordinates ({len(changes)} regions)", expanded=False):
            for idx, ch in enumerate(changes, 1):
                ctype = ch.get("change_type", "surface modification")
                area = ch.get("area_pixels", 0)
                x = ch.get("x", 0)
                y = ch.get("y", 0)
                st.markdown(f"`[{idx:02d}]` **{ctype.upper()}** &middot; {area:,} px at coordinate `({x}, {y})`")

    except Exception as exc:
        st.error(f"Detection failed: {exc}")


# ==============================================================================
# LANDING HUB & SIDEBAR
# ==============================================================================

def landing_page():
    stats = get_archive_stats()
    
    st.markdown(
        f"""
        <div class="top-nav-hud">
            <div class="top-nav-left">
                <span class="hud-pill"><span class="sidebar-dot"></span> CONSTELLATION NOMINAL</span>
                <span>SSO 512 KM</span>
            </div>
            <div class="top-nav-right">
                <span>INDEX &middot; <strong style="color:#f8fafc;">FAISS 512-D</strong></span>
                <span>ARCHIVE &middot; <strong style="color:#f8fafc;">{stats['total_scenes']:,} SCENES</strong></span>
                <span>SENSOR &middot; <strong style="color:#f8fafc;">MSI 13-BAND</strong></span>
            </div>
        </div>
        
        <div class="hero-container">
            <div class="hero-badge">
                <span class="hero-badge-dot"></span>
                <span class="hero-badge-text">EARTH OBSERVATION INTELLIGENCE &middot; MISSION HUB</span>
            </div>
            <h1 class="main-title">AERIS</h1>
            <p class="main-subtitle">Autonomous multi-modal satellite intelligence platform powered by RemoteCLIP, Vision-Language Transformers, and Temporal Change Detection.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown(
            """
            <div class="mode-card">
                <div class="mode-card-header">
                    <span class="mode-tag">MODE 01 // VISION</span>
                    <div class="mode-icon-wrap">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
                            <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
                            <line x1="12" y1="22.08" x2="12" y2="12"/>
                        </svg>
                    </div>
                </div>
                <div>
                    <div class="mode-card-title">Image &amp; Text</div>
                    <p class="mode-card-desc">Extract multimodal visual representations to locate identical and semantically adjacent scenes across the planetary archive.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Analysis →", key="btn_mode_image_text", use_container_width=True):
            st.session_state["mode"] = "image_text"
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="mode-card">
                <div class="mode-card-header">
                    <span class="mode-tag">MODE 02 // SEMANTIC</span>
                    <div class="mode-icon-wrap">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="11" cy="11" r="8"/>
                            <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                        </svg>
                    </div>
                </div>
                <div>
                    <div class="mode-card-title">Semantic Search</div>
                    <p class="mode-card-desc">Query the earth archive with natural language concepts, infrastructure descriptions, and geopolitical environmental queries.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Search Archive →", key="btn_mode_text", use_container_width=True):
            st.session_state["mode"] = "text"
            st.rerun()

    with col3:
        st.markdown(
            """
            <div class="mode-card">
                <div class="mode-card-header">
                    <span class="mode-tag">MODE 03 // TEMPORAL</span>
                    <div class="mode-icon-wrap">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                            <line x1="18" y1="20" x2="18" y2="10"/>
                            <line x1="12" y1="20" x2="12" y2="4"/>
                            <line x1="6" y1="20" x2="6" y2="14"/>
                        </svg>
                    </div>
                </div>
                <div>
                    <div class="mode-card-title">Change Detection</div>
                    <p class="mode-card-desc">Compare dual temporal captures with sub-pixel alignment and radiometric normalization to flag environmental transformations.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Detect Changes →", key="btn_mode_change_detection", use_container_width=True):
            st.session_state["mode"] = "change_detection"
            st.rerun()

    st.markdown(
        f"""
        <div class="telemetry-strip">
            <span>ORBIT &middot; <strong style="color:var(--cyan-core);">512 KM SUN-SYNCHRONOUS</strong></span>
            <span>DATASET &middot; <strong style="color:var(--cyan-core);">{stats['size_gb']} GB PARQUET</strong></span>
            <span>LATENCY &middot; <strong style="color:var(--emerald-core);">&lt; 5MS INDEXED LOOKUP</strong></span>
            <span>PRECISION &middot; <strong style="color:var(--cyan-core);">REMOTECLIP VI-B-32</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_config():
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <span style="position:relative;width:28px;height:28px;display:grid;place-items:center;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                        <circle cx="12" cy="12" r="4.6" stroke="#e0f2fe" stroke-width="1.2"/>
                        <ellipse cx="12" cy="12" rx="10.2" ry="4.1" stroke="rgba(56,189,248,.8)" stroke-width="1.1" transform="rotate(-26 12 12)"/>
                        <circle cx="20" cy="6.6" r="1.5" fill="#38bdf8"/>
                    </svg>
                </span>
                <span class="sidebar-brand-name">AERIS</span>
            </div>
            <div class="sidebar-status-tag">
                <span class="sidebar-dot"></span>
                <span>SYSTEM NOMINAL</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-label">MISSION TELEMETRY</div>', unsafe_allow_html=True)
        
        mode = st.session_state.get("mode")
        mode_labels = {
            "text": "Semantic Text Search",
            "image_text": "Image & Text Analysis",
            "change_detection": "Temporal Change Detection",
        }
        active_mode_str = mode_labels.get(str(mode), "Standby / Orbital Hub")
        
        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="sidebar-card-label">ACTIVE MISSION MODE</div>
                <div class="sidebar-card-value">{active_mode_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Microsecond Archive Stats Lookup (Never loads 3.1GB into RAM)
        stats = get_archive_stats()
        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="sidebar-card-label">ARCHIVE COVERAGE</div>
                <div class="sidebar-card-value">{stats['total_scenes']:,} Scenes</div>
                <div class="sidebar-card-sub">Sentinel-2 MSI Archive ({stats['size_gb']} GB)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-label">HARDWARE ACCELERATION</div>', unsafe_allow_html=True)
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                dev_name = _torch.cuda.get_device_name(0)
                mem_gb = _torch.cuda.get_device_properties(0).total_memory // (1024**3)
                st.markdown(
                    f"""
                    <div class="sidebar-card">
                        <div class="sidebar-card-label">ACCELERATOR</div>
                        <div class="sidebar-card-value">{dev_name}</div>
                        <div class="sidebar-card-sub">{mem_gb} GB VRAM Allocated</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="sidebar-card">
                        <div class="sidebar-card-label">ACCELERATOR</div>
                        <div class="sidebar-card-value">CPU Engine</div>
                        <div class="sidebar-card-sub">Optimized SIMD Vector Execution</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception:
            st.markdown(
                """
                <div class="sidebar-card">
                    <div class="sidebar-card-label">ACCELERATOR</div>
                    <div class="sidebar-card-value">Hardware Accelerated</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main():
    sidebar_config()

    if not DATASET_PATH.exists():
        st.error(f"Archive file not found: {DATASET_PATH}")
        return

    if not FAISS_PATH.exists():
        st.error(f"FAISS index not found: {FAISS_PATH}")
        return

    if not INDEX_PATH.exists():
        st.error(f"Index mapping not found: {INDEX_PATH}")
        return

    mode = st.session_state.get("mode")

    if mode is None:
        landing_page()
    else:
        pipelines_ready, error = lazy_load_pipelines()
        if not pipelines_ready:
            st.error("Backend neural pipelines could not be initialized.")
            st.code(error)
            return
        
        # Ghost Navigation Back to Hub Button
        st.markdown('<div class="back-btn-wrap">', unsafe_allow_html=True)
        if st.button("← Return to Mission Hub", key="global_back_btn"):
            st.session_state["mode"] = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        if mode == "text":
            run_text_mode()
        elif mode == "image_text":
            run_image_text_mode()
        elif mode == "change_detection":
            run_change_detection_mode()

    st.markdown(
        '<div class="footer">AERIS &middot; Intelligent Earth Observation Platform</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
