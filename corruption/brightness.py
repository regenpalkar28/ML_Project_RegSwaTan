"""
Brightness Corruption (Hendrycks & Dietterich benchmark).
Simulates severe day glare and sensor overexposure by elevating HSV luminance.
"""

from typing import Union
import cv2
import numpy as np
from PIL import Image


def brightness(image: Union[Image.Image, np.ndarray], severity: int = 1) -> Union[Image.Image, np.ndarray]:
    """
    Applies brightness elevation in HSV color space.

    Args:
        image: PIL Image or NumPy array (H, W, C) in range [0, 255].
        severity: Integer from 1 to 5.

    Returns:
        Corrupted image in the same format as input.
    """
    if not (1 <= severity <= 5):
        raise ValueError(f"Severity must be in range [1, 5], got {severity}")

    is_pil = isinstance(image, Image.Image)
    arr = np.array(image, dtype=np.uint8)

    # Shift factors in HSV value channel
    shifts = [0.08, 0.16, 0.24, 0.32, 0.42]
    shift = shifts[severity - 1]

    if arr.ndim == 3 and arr.shape[2] == 3:
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
        # Shift V channel (range 0..255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + shift * 255.0, 0, 255)
        corrupted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    else:
        corrupted = np.clip(arr.astype(np.float32) + shift * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(corrupted) if is_pil else corrupted
