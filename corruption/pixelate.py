"""
Pixelate (Mosaic Block) Corruption (Hendrycks & Dietterich benchmark).
Simulates severe downsampling and digital spatial resolution loss.
"""

from typing import Union
import cv2
import numpy as np
from PIL import Image


def pixelate(image: Union[Image.Image, np.ndarray], severity: int = 1) -> Union[Image.Image, np.ndarray]:
    """
    Applies spatial pixelation / mosaic effect via downsampling and nearest-neighbor upsampling.

    Args:
        image: PIL Image or NumPy array (H, W, C) in range [0, 255].
        severity: Integer from 1 (mild) to 5 (heavy blockiness).

    Returns:
        Corrupted image in the same format as input.
    """
    if not (1 <= severity <= 5):
        raise ValueError(f"Severity must be in range [1, 5], got {severity}")

    is_pil = isinstance(image, Image.Image)
    arr = np.array(image, dtype=np.uint8)
    h, w = arr.shape[:2]

    # Downsampling scale factors (lower = larger mosaic blocks)
    downscale_factors = [0.80, 0.65, 0.50, 0.35, 0.20]
    scale = downscale_factors[severity - 1]

    small_w = max(4, int(np.round(w * scale)))
    small_h = max(4, int(np.round(h * scale)))

    # Downsample using area interpolation
    downsampled = cv2.resize(arr, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    # Upsample using nearest neighbor (creates distinct mosaic pixel blocks)
    pixelated = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_NEAREST)

    return Image.fromarray(pixelated) if is_pil else pixelated
