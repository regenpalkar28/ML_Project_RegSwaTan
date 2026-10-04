"""
Gaussian Noise Corruption (Hendrycks & Dietterich benchmark).
Simulates low-light sensor noise across 5 severity levels.
"""

from typing import Union
import numpy as np
from PIL import Image


def gaussian_noise(image: Union[Image.Image, np.ndarray], severity: int = 1) -> Union[Image.Image, np.ndarray]:
    """
    Applies additive Gaussian white noise to an image.
    
    Mathematical formulation:
        I_corrupted = clip(I + N(0, sigma^2), 0, 1)

    Args:
        image: PIL Image or NumPy array (H, W, C) in range [0, 255].
        severity: Integer from 1 (mild) to 5 (extreme).

    Returns:
        Corrupted image in the same format as input.
    """
    if not (1 <= severity <= 5):
        raise ValueError(f"Severity must be in range [1, 5], got {severity}")

    is_pil = isinstance(image, Image.Image)
    arr = np.array(image, dtype=np.float32) / 255.0

    # Severity scale calibrated for traffic sign recognition (Hendrycks CIFAR-C / ImageNet-C scale)
    sigmas = [0.04, 0.06, 0.08, 0.10, 0.14]
    sigma = sigmas[severity - 1]

    noise = np.random.normal(loc=0.0, scale=sigma, size=arr.shape)
    corrupted = np.clip(arr + noise, 0.0, 1.0) * 255.0
    corrupted = corrupted.astype(np.uint8)

    return Image.fromarray(corrupted) if is_pil else corrupted
