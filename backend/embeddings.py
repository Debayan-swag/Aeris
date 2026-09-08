"""
TerraWatch AI - Multimodal Embedding Service

Generates normalized embeddings for:

1. Satellite images
2. Natural-language text queries

The embeddings are designed for semantic retrieval using FAISS.
"""

import sys
from pathlib import Path
from typing import List, Union

# Ensure project root is in sys.path for direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image

from backend.config import settings


class EmbeddingService:

    def __init__(
        self,
        model_name: str = settings.VISION_EMBEDDING_MODEL,
        pretrained: str = settings.VISION_EMBEDDING_PRETRAINED,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained

        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = None

    # =========================================================
    # MODEL LOADING
    # =========================================================

    def _lazy_load(self) -> None:
        """
        Load the OpenCLIP model only when it is needed.
        """
        if self._model is not None:
            return

        try:
            import torch
            import open_clip  # type: ignore

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            print("\nLoading embedding model...")
            print(f"Model: {self.model_name}")
            print(f"Pretrained: {self.pretrained}")
            print(f"Device: {self._device}")

            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self._device,
            )

            self._model = model.eval()
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self.model_name)

            print("Embedding model loaded successfully!")

        except Exception as error:
            raise RuntimeError(
                "Failed to load embedding model.\n"
                f"Model: {self.model_name}\n"
                f"Error: {error}"
            ) from error

    # =========================================================
    # TEXT EMBEDDING
    # =========================================================

    def embed_text(
        self,
        text: Union[str, List[str]],
    ) -> np.ndarray:
        """
        Generate normalized embeddings for text.

        Returns:
            numpy.ndarray
            Shape:
                (number_of_texts, embedding_dimension)
        """
        self._lazy_load()

        import torch

        if isinstance(text, str):
            text = [text]

        if not text:
            raise ValueError("Text input cannot be empty.")

        tokens = self._tokenizer(text).to(self._device)

        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(
                dim=-1,
                keepdim=True,
            )

        return features.cpu().numpy().astype(np.float32)

    # =========================================================
    # IMAGE EMBEDDING
    # =========================================================

    def embed_image(
        self,
        image: Union[Image.Image, List[Image.Image]],
    ) -> np.ndarray:
        """
        Generate normalized embeddings for one or more images.

        Returns:
            numpy.ndarray
            Shape:
                (number_of_images, embedding_dimension)
        """
        self._lazy_load()

        import torch

        if isinstance(image, Image.Image):
            images = [image]
        else:
            images = image

        if not images:
            raise ValueError("Image input cannot be empty.")

        tensors = torch.stack(
            [self._preprocess(img.convert("RGB")) for img in images]
        ).to(self._device)

        with torch.no_grad():
            features = self._model.encode_image(tensors)
            features = features / features.norm(
                dim=-1,
                keepdim=True,
            )

        return features.cpu().numpy().astype(np.float32)

    # =========================================================
    # IMAGE PATH EMBEDDING
    # =========================================================

    def embed_image_path(
        self,
        image_path: Union[str, Path],
    ) -> np.ndarray:
        """
        Generate an embedding directly from an image file path.
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as image:
            return self.embed_image(image.copy())

    # =========================================================
    # EMBEDDING INFORMATION
    # =========================================================

    def get_embedding_dimension(self) -> int:
        """
        Return the embedding dimension.
        """
        self._lazy_load()

        # Generate one small test embedding
        embedding = self.embed_text("satellite image")
        return embedding.shape[1]


# =============================================================
# GLOBAL EMBEDDING SERVICE
# =============================================================

embedding_service = EmbeddingService()


# =============================================================
# TEST EMBEDDING SERVICE
# =============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TERRAWATCH AI - EMBEDDING SERVICE TEST")
    print("=" * 60)

    test_text = "satellite image showing an urban area with buildings and roads"

    print("\nGenerating text embedding...")
    embedding = embedding_service.embed_text(test_text)

    print("\nEmbedding generated successfully!")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dimension: {embedding.shape[1]}")
    print(f"Data type: {embedding.dtype}")
    print("\nFirst 10 embedding values:")
    print(embedding[0][:10])

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
