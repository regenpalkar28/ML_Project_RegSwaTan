"""
Zoom Blur (Radial Blur) Corruption (Hendrycks & Dietterich benchmark).
Simulates high-speed forward vehicle approach toward a sign during exposure.
"""

from typing import Union
import cv2
import numpy as np
from PIL import Image


def _clipped_zoom(img: np.ndarray, zoom_factor: float) -> np.ndarray:
    """Scale image from center and clip/pad to original dimensions."""
    h, w = img.shape[:2]
    new_h, new_w = int(np.round(h * zoom_factor)), int(np.round(w * zoom_factor))

    # Resize image
    zoomed = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Crop center back to (h, w)
    top = (new_h - h) // 2
    left = (new_w - w) // 2
    cropped = zoomed[top:top + h, left:left + w]

    # Ensure shape matches exactly
    if cropped.shape[:2] != (h, w):
        cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return cropped


def zoom_blur(image: Union[Image.Image, np.ndarray], severity: int = 1) -> Union[Image.Image, np.ndarray]:
    """
    Applies zoom blur by averaging successive center-scaled crops.

    Args:
        image: PIL Image or NumPy array (H, W, C) in range [0, 255].
        severity: Integer from 1 to 5.

    Returns:
        Corrupted image in the same format as input.
    """
    if not (1 <= severity <= 5):
        raise ValueError(f"Severity must be in range [1, 5], got {severity}")

    is_pil = isinstance(image, Image.Image)
    arr = np.array(image, dtype=np.float32) / 255.0

    # Zoom scale ranges across severities (from Hendrycks CIFAR-C / ImageNet-C)
    zoom_factors_list = [
        np.arange(1.0, 1.06, 0.01),
        np.arange(1.0, 1.11, 0.01),
        np.arange(1.0, 1.16, 0.01),
        np.arange(1.0, 1.21, 0.01),
        np.arange(1.0, 1.28, 0.015),
    ]
    factors = zoom_factors_list[severity - 1]

    accum = np.zeros_like(arr)
    for z in factors:
        accum += _clipped_zoom(arr, float(z))

    blended = (arr + accum) / (len(factors) + 1)
    corrupted = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(corrupted) if is_pil else corrupted
