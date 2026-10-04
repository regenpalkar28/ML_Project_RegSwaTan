"""
Snow Corruption (Hendrycks & Dietterich benchmark).
Simulates falling snowflakes with directional motion blur and surface snow whitening.
"""

from typing import Union
import cv2
import numpy as np
from PIL import Image

from .motion_blur import _motion_blur_kernel


def snow(image: Union[Image.Image, np.ndarray], severity: int = 1) -> Union[Image.Image, np.ndarray]:
    """
    Applies snow weather corruption.
    
    Combines:
    1. A sparse snowflake particle layer with directional wind motion blur.
    2. Atmospheric scene whitening (light scattering off snow crystals).

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
    h, w = arr.shape[:2]

    # Parameters: (particle_mean, particle_std, threshold, blur_size, blur_sigma, whitening_factor)
    params = [
        (0.1, 0.2, 0.70, 5, 2.0, 0.92),
        (0.15, 0.25, 0.65, 7, 2.5, 0.85),
        (0.20, 0.30, 0.60, 9, 3.0, 0.78),
        (0.28, 0.35, 0.55, 11, 4.0, 0.70),
        (0.35, 0.40, 0.50, 13, 5.0, 0.60),
    ]
    p_mean, p_std, thresh, k_size, k_sigma, white_factor = params[severity - 1]

    # 1. Generate snow particles
    snow_layer = np.random.normal(loc=p_mean, scale=p_std, size=(h, w)).astype(np.float32)
    snow_layer = np.where(snow_layer > thresh, snow_layer, 0.0)

    # 2. Apply directional motion blur (falling at angle between -65 and -35 degrees)
    angle = float(np.random.uniform(-65.0, -35.0))
    blur_kernel = _motion_blur_kernel(size=k_size, angle=angle)
    blurred_snow = cv2.filter2D(snow_layer, -1, blur_kernel)
    blurred_snow = np.clip(blurred_snow, 0.0, 1.0)
    if arr.ndim == 3:
        blurred_snow = blurred_snow[..., np.newaxis]

    # 3. Ground / object whitening (contrast reduction towards white)
    if arr.ndim == 3:
        gray = cv2.cvtColor((arr * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        whitened = np.maximum(arr, gray[..., np.newaxis] * 1.3 + 0.3)
    else:
        whitened = np.maximum(arr, arr * 1.3 + 0.3)

    blended = white_factor * arr + (1.0 - white_factor) * whitened
    corrupted = np.clip((blended + blurred_snow) * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(corrupted) if is_pil else corrupted
