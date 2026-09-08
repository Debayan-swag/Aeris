"""
config.py
Configuration settings for TerraWatch backend.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    PROJECT_NAME: str = "TerraWatch"
    API_V1_STR: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_PARQUET_PATH: Path = BASE_DIR / "data" / "raw" / "sentinel2.parquet"
    PROCESSED_IMAGES_DIR: Path = BASE_DIR / "data" / "processed" / "images"
    PROCESSED_METADATA_DIR: Path = BASE_DIR / "data" / "processed" / "metadata"
    NEW_IMAGES_DIR: Path = BASE_DIR / "data" / "new_images"

    DATABASE_PATH: Path = BASE_DIR / "database" / "terrawatch.db"
    INDEXES_DIR: Path = BASE_DIR / "indexes"
    FAISS_INDEX_PATH: Path = BASE_DIR / "indexes" / "satellite.faiss"
    EMBEDDINGS_PATH: Path = BASE_DIR / "indexes" / "embeddings.npy"
    PATHS_MAP_PATH: Path = BASE_DIR / "indexes" / "image_paths.json"
    IMAGE_PATHS_PATH: Path = BASE_DIR / "indexes" / "image_paths.json"
    IMAGE_IDS_PATH: Path = BASE_DIR / "indexes" / "image_ids.json"
    MODELS_DIR: Path = BASE_DIR / "models"

    # Model defaults
    VISION_EMBEDDING_MODEL: str = "ViT-B-32"
    VISION_EMBEDDING_PRETRAINED: str = "laion2b_s34b_b79k"
    VLM_MODEL_ID: str = "Qwen/Qwen2-VL-7B-Instruct"

    # Search & Retrieval
    DEFAULT_TOP_K: int = 10
    SIMILARITY_THRESHOLD: float = 0.25

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
