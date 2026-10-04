"""
Elastic Deformation Corruption (Hendrycks & Dietterich benchmark).
Simulates physical non-rigid warping, lens distortions, and sensor heat shimmer.
"""

from typing import Union
import cv2
import numpy as np
from PIL import Image


def elastic_transform(image: Union[Image.Image, np.ndarray], severity: int = 1) -> Union[Image.Image, np.ndarray]:
    """
    Applies non-rigid elastic deformations based on Simard et al. (2003).
    
    Generates smooth random displacement vector fields (dx, dy) filtered with a Gaussian kernel
    and remaps image coordinates using bilinear interpolation.

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
    h, w = arr.shape[:2]

    # Parameters: (alpha = displacement magnitude, sigma = smoothness, affine_jitter)
    # Scaled proportionally to image dimensions
    dim = min(h, w)
    params = [
        (dim * 0.40, dim * 0.08, dim * 0.04),
        (dim * 0.70, dim * 0.09, dim * 0.06),
        (dim * 1.10, dim * 0.10, dim * 0.08),
        (dim * 1.60, dim * 0.11, dim * 0.10),
        (dim * 2.20, dim * 0.12, dim * 0.12),
    ]
    alpha, sigma, affine_jitter = params[severity - 1]

    # 1. Mild affine perturbation
    center = (w // 2, h // 2)
    pts1 = np.float32([[center[0], center[1]], [center[0] + 10, center[1]], [center[0], center[1] + 10]])
    jitter = np.random.uniform(-affine_jitter, affine_jitter, size=pts1.shape).astype(np.float32)
    pts2 = pts1 + jitter
    m_affine = cv2.getAffineTransform(pts1, pts2)
    arr_warped = cv2.warpAffine(arr, m_affine, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # 2. Smooth non-rigid vector displacement field
    dx = np.random.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32)
    dy = np.random.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32)

    ksize = int(np.round(sigma * 4)) | 1  # Ensure odd kernel size
    ksize = max(3, ksize)
    dx = cv2.GaussianBlur(dx, (ksize, ksize), sigmaX=sigma, sigmaY=sigma) * alpha
    dy = cv2.GaussianBlur(dy, (ksize, ksize), sigmaX=sigma, sigmaY=sigma) * alpha

    # 3. Create grid and remap
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)

    corrupted = cv2.remap(arr_warped, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return Image.fromarray(corrupted) if is_pil else corrupted
