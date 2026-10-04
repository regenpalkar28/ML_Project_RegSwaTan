"""
Contrast Reduction Corruption (Hendrycks & Dietterich benchmark).
Simulates low dynamic range and washed-out visibility by scaling variance around mean luminance.
"""

from typing import Union
import numpy as np
from PIL import Image


def contrast(image: Union[Image.Image, np.ndarray], severity: int = 1) -> Union[Image.Image, np.ndarray]:
    """
    Applies contrast attenuation towards mean luminance.
    
    Mathematical formulation:
        I_corrupted = clip((I - mean(I)) * c + mean(I), 0, 1)
        where smaller c produces heavier contrast loss.

    Args:
        image: PIL Image or NumPy array (H, W, C) in range [0, 255].
        severity: Integer from 1 (mild attenuation) to 5 (heavy attenuation).

    Returns:
        Corrupted image in the same format as input.
    """
    if not (1 <= severity <= 5):
        raise ValueError(f"Severity must be in range [1, 5], got {severity}")

    is_pil = isinstance(image, Image.Image)
    arr = np.array(image, dtype=np.float32) / 255.0

    # Scaling multipliers: lower value = lower contrast
    scales = [0.65, 0.48, 0.35, 0.22, 0.10]
    c = scales[severity - 1]

    means = np.mean(arr, axis=(0, 1), keepdims=True)
    corrupted = np.clip((arr - means) * c + means, 0.0, 1.0) * 255.0
    corrupted = corrupted.astype(np.uint8)

    return Image.fromarray(corrupted) if is_pil else corrupted
