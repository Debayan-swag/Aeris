"""
Aeris Change Detection Visualization Module
Provides utility functions for computing difference masks, heatmaps, and overlays.
"""
import cv2
import numpy as np
from PIL import Image

def create_difference_mask(image1, image2):
    """
    Computes difference visualizations between two images.
    
    Args:
        image1: First image (PIL.Image or numpy array)
        image2: Second image (PIL.Image or numpy array)
        
    Returns:
        tuple: (binary_mask, heatmap_rgb, overlay_rgb) as numpy arrays or PIL Images
    """
    # Convert PIL Image to RGB numpy array if needed
    if isinstance(image1, Image.Image):
        arr1 = np.array(image1.convert("RGB"))
    else:
        arr1 = np.array(image1)
        if len(arr1.shape) == 2:
            arr1 = cv2.cvtColor(arr1, cv2.COLOR_GRAY2RGB)
        elif arr1.shape[2] == 4:
            arr1 = cv2.cvtColor(arr1, cv2.COLOR_RGBA2RGB)

    if isinstance(image2, Image.Image):
        arr2 = np.array(image2.convert("RGB"))
    else:
        arr2 = np.array(image2)
        if len(arr2.shape) == 2:
            arr2 = cv2.cvtColor(arr2, cv2.COLOR_GRAY2RGB)
        elif arr2.shape[2] == 4:
            arr2 = cv2.cvtColor(arr2, cv2.COLOR_RGBA2RGB)

    # Ensure identical dimensions
    h1, w1 = arr1.shape[:2]
    h2, w2 = arr2.shape[:2]
    if (h1, w1) != (h2, w2):
        arr2 = cv2.resize(arr2, (w1, h1), interpolation=cv2.INTER_LANCZOS4)

    # Calculate absolute difference
    diff = cv2.absdiff(arr1, arr2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    
    # Smooth to suppress sensor noise
    blurred = cv2.GaussianBlur(gray_diff, (5, 5), 0)

    # Adaptive or Otsu threshold for crisp binary mask
    _, binary_mask = cv2.threshold(blurred, 32, 255, cv2.THRESH_BINARY)
    
    # Normalized difference for thermal/scientific heatmap
    norm_diff = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_bgr = cv2.applyColorMap(norm_diff.astype(np.uint8), cv2.COLORMAP_TURBO)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # High-tech composite overlay
    overlay_rgb = cv2.addWeighted(arr1, 0.65, heatmap_rgb, 0.35, 0)

    return binary_mask, heatmap_rgb, overlay_rgb
