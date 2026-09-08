"""
change_detection.py
Satellite image difference and change detection algorithms.
Supports SSIM difference, absolute difference, and threshold mask generation.
"""

from typing import Dict, Tuple
import numpy as np
from PIL import Image


def compute_image_difference(
    img_before: Image.Image,
    img_after: Image.Image,
    threshold: float = 0.15,
) -> Tuple[float, Image.Image, Dict[str, float]]:
    """
    Computes structural/pixel change between two satellite images of the same area.
    Returns:
        change_score: float (0.0 to 1.0)
        mask_image: PIL Image showing highlighted change regions
        metrics: dict of detailed metrics (mean difference, changed pixel percentage)
    """
    # Resize after to match before if dimensions differ
    if img_before.size != img_after.size:
        img_after = img_after.resize(img_before.size, Image.Resampling.BILINEAR)

    arr1 = np.array(img_before.convert("RGB"), dtype=np.float32) / 255.0
    arr2 = np.array(img_after.convert("RGB"), dtype=np.float32) / 255.0

    # Pixel difference
    diff = np.abs(arr1 - arr2)
    diff_gray = np.mean(diff, axis=2)

    # Change mask
    mask_binary = (diff_gray > threshold).astype(np.uint8) * 255
    change_ratio = float(np.count_nonzero(mask_binary) / mask_binary.size)

    # Visual heatmap overlay
    overlay = np.zeros_like(arr1)
    overlay[..., 0] = diff_gray * 255  # Red channel shows difference
    mask_pil = Image.fromarray(mask_binary, mode="L")

    metrics = {
        "mean_diff": float(np.mean(diff_gray)),
        "max_diff": float(np.max(diff_gray)),
        "changed_pixel_ratio": change_ratio,
    }

    return change_ratio, mask_pil, metrics
