"""
RemoteCLIP Embedding Service

Loads and uses RemoteCLIP (remote sensing specialized CLIP) for generating
embeddings of satellite images and text queries.

RemoteCLIP is trained specifically on remote sensing imagery and outperforms
standard CLIP models on satellite image tasks.

Reference: https://github.com/ChenDelong1999/RemoteCLIP
Paper: "RemoteCLIP: A Vision Language Foundation Model for Remote Sensing" (IEEE TGRS)
"""

import sys
from pathlib import Path
from typing import List, Union

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image


class RemoteCLIPEmbeddingService:
    """
    Embedding service for RemoteCLIP model.
    
    Compatible with OpenCLIP API for easy comparison.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        checkpoint_path: str = None,
        cache_dir: str = None,
    ) -> None:
        """
        Initialize RemoteCLIP service.
        
        Args:
            model_name: Model architecture ('RN50', 'ViT-B-32', or 'ViT-L-14')
            checkpoint_path: Path to RemoteCLIP checkpoint (auto-downloaded if None)
            cache_dir: Directory to cache downloaded checkpoints
        """
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.cache_dir = cache_dir or str(ROOT_DIR / "checkpoints" / "remoteclip")
        
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = None
    
    # =========================================================
    # CHECKPOINT DOWNLOAD
    # =========================================================
    
    def _download_checkpoint(self) -> str:
        """
        Download RemoteCLIP checkpoint from HuggingFace.
        
        Returns:
            Path to downloaded checkpoint
        """
        try:
            from huggingface_hub import hf_hub_download
            
            print(f"\nDownloading RemoteCLIP-{self.model_name} checkpoint...")
            print(f"This may take a few minutes on first run.")
            
            checkpoint_path = hf_hub_download(
                repo_id="chendelong/RemoteCLIP",
                filename=f"RemoteCLIP-{self.model_name}.pt",
                cache_dir=self.cache_dir,
            )
            
            print(f"✓ Checkpoint downloaded to: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as error:
            raise RuntimeError(
                f"Failed to download RemoteCLIP checkpoint.\n"
                f"Model: {self.model_name}\n"
                f"Error: {error}\n\n"
                f"Please check your internet connection and try again."
            ) from error
    
    # =========================================================
    # MODEL LOADING
    # =========================================================
    
    def _lazy_load(self) -> None:
        """
        Load RemoteCLIP model only when needed.
        """
        if self._model is not None:
            return
        
        try:
            import torch
            import open_clip
            
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print("\n" + "=" * 70)
            print("LOADING REMOTECLIP")
            print("=" * 70)
            print(f"\nModel: RemoteCLIP-{self.model_name}")
            print(f"Device: {self._device}")
            
            # Download checkpoint if not provided
            if self.checkpoint_path is None:
                self.checkpoint_path = self._download_checkpoint()
            
            # Initialize OpenCLIP model architecture
            print(f"\nInitializing {self.model_name} architecture...")
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                device=self._device,
            )
            
            # Load RemoteCLIP weights
            print(f"Loading RemoteCLIP weights from checkpoint...")
            ckpt = torch.load(self.checkpoint_path, map_location=self._device)
            message = model.load_state_dict(ckpt)
            print(f"Load status: {message}")
            
            # Set to evaluation mode
            self._model = model.eval()
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            
            print("\n✓ RemoteCLIP loaded successfully!")
            print("=" * 70 + "\n")
            
        except Exception as error:
            raise RuntimeError(
                f"Failed to load RemoteCLIP model.\n"
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
        
        Args:
            text: Single string or list of strings
        
        Returns:
            numpy.ndarray of shape (num_texts, embedding_dimension)
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
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy().astype(np.float32)
    
    # =========================================================
    # IMAGE EMBEDDING
    # =========================================================
    
    def embed_image(
        self,
        image: Union[Image.Image, List[Image.Image]],
    ) -> np.ndarray:
        """
        Generate normalized embeddings for images.
        
        Args:
            image: Single PIL Image or list of PIL Images
        
        Returns:
            numpy.ndarray of shape (num_images, embedding_dimension)
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
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy().astype(np.float32)
    
    # =========================================================
    # IMAGE PATH EMBEDDING
    # =========================================================
    
    def embed_image_path(
        self,
        image_path: Union[str, Path],
    ) -> np.ndarray:
        """
        Generate embedding directly from image file path.
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
        """Return the embedding dimension."""
        self._lazy_load()
        
        # Generate test embedding to get dimension
        embedding = self.embed_text("test")
        return embedding.shape[1]


# =============================================================
# GLOBAL SERVICE INSTANCE
# =============================================================

remoteclip_service = RemoteCLIPEmbeddingService()


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("REMOTECLIP EMBEDDING SERVICE TEST")
    print("=" * 70)
    
    # Test text embedding
    test_text = "satellite image showing urban area with buildings"
    print(f"\nTest query: \"{test_text}\"")
    
    embedding = remoteclip_service.embed_text(test_text)
    
    print(f"\n✓ Embedding generated successfully!")
    print(f"  Shape: {embedding.shape}")
    print(f"  Dimension: {embedding.shape[1]}")
    print(f"  Data type: {embedding.dtype}")
    print(f"  First 10 values: {embedding[0][:10]}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70 + "\n")
