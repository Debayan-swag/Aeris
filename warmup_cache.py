"""
Cache Warmup Script for Aeris
Pre-loads heavy resources to speed up first-time Streamlit startup
"""
import sys
from pathlib import Path

print("Warming up Aeris cache...")
print("-" * 50)

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

try:
    # 1. Load models (heaviest operation)
    print("1/4 Loading AI models...")
    from image_text import remoteclip_model, remoteclip_tokenizer
    print("RemoteCLIP loaded")
    
    # 2. Load FAISS index
    print("2/4 Loading search index...")
    import faiss
    import pandas as pd
    FAISS_PATH = PROJECT_DIR / "remoteclip_embeddings" / "remoteclip.faiss"
    INDEX_PATH = PROJECT_DIR / "remoteclip_embeddings" / "remoteclip_index.parquet"
    
    vector_store = faiss.read_index(str(FAISS_PATH))
    index_mapping = pd.read_parquet(INDEX_PATH, engine='pyarrow')
    print(f"FAISS index loaded ({len(index_mapping):,} embeddings)")
    
    # 3. Load dataset metadata only (not full dataset)
    print("3/4 Loading dataset metadata...")
    DATASET_PATH = PROJECT_DIR / "sentinel-2-processed.parquet"
    df_info = pd.read_parquet(DATASET_PATH, columns=['image_name'])
    print(f"Dataset ready ({len(df_info):,} images)")
    
    # 4. Test text pipeline
    print("4/4 Testing text generation...")
    from transformers import pipeline
    text_gen = pipeline("text-generation", model="distilbert/distilgpt2", device=-1)
    test = text_gen("Test", max_new_tokens=5, do_sample=False)
    print("   Text generator ready")
    
    print("-" * 50)
    print("Cache warmup complete!")
    print("Streamlit will now load much faster!")
    print("-" * 50)
    
except Exception as e:
    print(f"\nWarmup failed: {e}")
    print("Streamlit will still work, but may load slower on first run.")
    sys.exit(1)
