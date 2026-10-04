"""
Impulse Noise (Salt and Pepper) Corruption (Hendrycks & Dietterich benchmark).
Simulates dead sensor pixels and transmission bit corruption across 5 severity levels.
"""

from typing import Union
import numpy as np
from PIL import Image


def impulse_noise(image: Union[Image.Image, np.ndarray], severity: int = 1) -> Union[Image.Image, np.ndarray]:
    """
    Applies salt-and-pepper impulse noise by randomly setting pixels to minimum (0) or maximum (255).

    Args:
        image: PIL Image or NumPy array (H, W, C) in range [0, 255].
        severity: Integer from 1 (mild, 1.5% pixels) to 5 (severe, 15% pixels).

    Returns:
        Corrupted image in the same format as input.
    """
    if not (1 <= severity <= 5):
        raise ValueError(f"Severity must be in range [1, 5], got {severity}")

    is_pil = isinstance(image, Image.Image)
    arr = np.array(image, dtype=np.uint8).copy()

    # Fraction of corrupted pixels (salt & pepper)
    amounts = [0.015, 0.03, 0.06, 0.10, 0.15]
    amount = amounts[severity - 1]

    h, w = arr.shape[:2]
    num_corrupt = int(h * w * amount)

    # Random coordinate selection
    y_coords = np.random.randint(0, h, size=num_corrupt)
    x_coords = np.random.randint(0, w, size=num_corrupt)

    # Half salt (255), half pepper (0)
    split = num_corrupt // 2
    # Salt
    arr[y_coords[:split], x_coords[:split]] = 255
    # Pepper
    arr[y_coords[split:], x_coords[split:]] = 0

    return Image.fromarray(arr) if is_pil else arr
