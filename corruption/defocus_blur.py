"""
Defocus Blur Corruption (Hendrycks & Dietterich benchmark).
Simulates optical lens out-of-focus blur using disc point spread function (PSF).
"""

from typing import Union
import cv2
import numpy as np
from PIL import Image


def _disk_kernel(radius: float, alias_blur: float = 0.5) -> np.ndarray:
    """Generate an anti-aliased disk point spread function kernel."""
    r = int(np.ceil(radius))
    d = 2 * r + 1
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    dist = np.sqrt(x * x + y * y)
    kernel = (dist <= radius).astype(np.float32)
    # Anti-alias with mild Gaussian blur
    kernel = cv2.GaussianBlur(kernel, (3, 3), sigmaX=alias_blur, sigmaY=alias_blur)
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel /= kernel_sum
    return kernel


def defocus_blur(image: Union[Image.Image, np.ndarray], severity: int = 1) -> Union[Image.Image, np.ndarray]:
    """
    Applies defocus blur simulating camera lens focus error.

    Args:
        image: PIL Image or NumPy array (H, W, C) in range [0, 255].
        severity: Integer from 1 (mild) to 5 (strong).

    Returns:
        Corrupted image in the same format as input.
    """
    if not (1 <= severity <= 5):
        raise ValueError(f"Severity must be in range [1, 5], got {severity}")

    is_pil = isinstance(image, Image.Image)
    arr = np.array(image, dtype=np.float32)

    # Disk radii parameters
    radii = [0.8, 1.2, 1.8, 2.6, 3.6]
    r = radii[severity - 1]

    kernel = _disk_kernel(r, alias_blur=0.4)

    # Filter each color channel independently
    if arr.ndim == 3:
        blurred = np.zeros_like(arr)
        for c in range(arr.shape[2]):
            blurred[:, :, c] = cv2.filter2D(arr[:, :, c], -1, kernel)
    else:
        blurred = cv2.filter2D(arr, -1, kernel)

    corrupted = np.clip(blurred, 0, 255).astype(np.uint8)
    return Image.fromarray(corrupted) if is_pil else corrupted
