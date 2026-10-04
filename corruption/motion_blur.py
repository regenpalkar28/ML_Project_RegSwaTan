"""
Motion Blur Corruption (Hendrycks & Dietterich benchmark).
Simulates vehicle or camera motion during image capture.
"""

from typing import Union
import cv2
import numpy as np
from PIL import Image


def _motion_blur_kernel(size: int, angle: float) -> np.ndarray:
    """Generate a directional motion point spread function (PSF) kernel."""
    size = max(3, size)
    if size % 2 == 0:
        size += 1

    kernel = np.zeros((size, size), dtype=np.float32)
    mid = size // 2
    kernel[mid, :] = 1.0

    # Rotate kernel to target angle
    rot_mat = cv2.getRotationMatrix2D((mid, mid), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (size, size))
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel /= kernel_sum
    return kernel


def motion_blur(image: Union[Image.Image, np.ndarray], severity: int = 1) -> Union[Image.Image, np.ndarray]:
    """
    Applies motion blur along a random trajectory angle.

    Args:
        image: PIL Image or NumPy array (H, W, C) in range [0, 255].
        severity: Integer from 1 to 5.

    Returns:
        Corrupted image in the same format as input.
    """
    if not (1 <= severity <= 5):
        raise ValueError(f"Severity must be in range [1, 5], got {severity}")

    is_pil = isinstance(image, Image.Image)
    arr = np.array(image, dtype=np.float32)

    # Kernel length parameters across severities
    kernel_sizes = [3, 5, 7, 9, 13]
    size = kernel_sizes[severity - 1]
    angle = float(np.random.uniform(-45.0, 45.0))

    kernel = _motion_blur_kernel(size=size, angle=angle)

    if arr.ndim == 3:
        blurred = np.zeros_like(arr)
        for c in range(arr.shape[2]):
            blurred[:, :, c] = cv2.filter2D(arr[:, :, c], -1, kernel)
    else:
        blurred = cv2.filter2D(arr, -1, kernel)

    corrupted = np.clip(blurred, 0, 255).astype(np.uint8)
    return Image.fromarray(corrupted) if is_pil else corrupted
