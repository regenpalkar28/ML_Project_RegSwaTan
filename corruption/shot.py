"""
Shot Noise (Poisson Noise) Corruption (Hendrycks & Dietterich benchmark).
Simulates photon count fluctuations caused by sensor quantum limits.
"""

from typing import Union
import numpy as np
from PIL import Image


def shot_noise(image: Union[Image.Image, np.ndarray], severity: int = 1) -> Union[Image.Image, np.ndarray]:
    """
    Applies Shot Noise (Poisson distributed photon fluctuation noise).
    
    Mathematical formulation:
        I_corrupted = clip(Poisson(I * c) / c, 0, 1)
        where smaller c represents fewer detected photons per pixel.

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

    # Photon count scaling parameters: smaller c -> more prominent shot noise
    photons = [250.0, 100.0, 50.0, 25.0, 12.0]
    c = photons[severity - 1]

    # Poisson noise generation
    noisy = np.random.poisson(np.maximum(0.0, arr * c)) / c
    corrupted = np.clip(noisy, 0.0, 1.0) * 255.0
    corrupted = corrupted.astype(np.uint8)

    return Image.fromarray(corrupted) if is_pil else corrupted
