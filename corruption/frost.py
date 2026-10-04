"""
Frost Corruption (Hendrycks & Dietterich benchmark).
Simulates crystalline ice/frost build-up on the camera lens or windshield.
"""

from pathlib import Path
from typing import Union
import cv2
import numpy as np
from PIL import Image


def _generate_procedural_frost(h: int, w: int) -> np.ndarray:
    """
    Procedurally synthesizes crystalline ice patterns without requiring external image assets.
    Combines high-frequency Voronoi-like crystal noise and dendritic edge filters.
    """
    scale = max(h, w)
    # Random seed points for crystal nucleation
    num_seeds = max(15, scale // 2)
    y_seeds = np.random.randint(0, h, size=num_seeds)
    x_seeds = np.random.randint(0, w, size=num_seeds)

    # Compute Euclidean distance map to nearest crystal seed
    seed_grid = np.zeros((h, w), dtype=np.uint8)
    seed_grid[y_seeds, x_seeds] = 255
    dist = cv2.distanceTransform(255 - seed_grid, cv2.DIST_L2, 5)

    # High frequency crystalline ridges
    ridges = np.sin(dist * 0.8) ** 2
    crystals = (ridges * 255.0).astype(np.uint8)

    # Blend with fine high-frequency noise for frosty ice texture
    fine_noise = np.random.normal(128, 40, (h, w)).astype(np.float32)
    frost_texture = 0.6 * crystals.astype(np.float32) + 0.4 * fine_noise
    frost_texture = cv2.GaussianBlur(frost_texture, (3, 3), 0.5)

    frost_norm = (frost_texture - frost_texture.min()) / (frost_texture.max() - frost_texture.min() + 1e-6)
    # Convert to 3-channel RGB (light ice-blue tint)
    frost_rgb = np.zeros((h, w, 3), dtype=np.float32)
    frost_rgb[:, :, 0] = frost_norm * 220.0  # R
    frost_rgb[:, :, 1] = frost_norm * 240.0  # G
    frost_rgb[:, :, 2] = frost_norm * 255.0  # B (bluish)
    return frost_rgb


def frost(image: Union[Image.Image, np.ndarray], severity: int = 1) -> Union[Image.Image, np.ndarray]:
    """
    Applies frost corruption over the image across 5 severity levels.

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
    h, w = arr.shape[:2]

    # Blend weights (image_weight, frost_weight) from Hendrycks benchmark
    weights = [
        (0.90, 0.25),
        (0.80, 0.40),
        (0.70, 0.55),
        (0.60, 0.70),
        (0.48, 0.85),
    ]
    img_w, frost_w = weights[severity - 1]

    # Check for local frost template images first
    frost_pattern = None
    frost_files = list(Path(__file__).parent.glob("frost*.png")) + list(Path(__file__).parent.glob("frost*.jpg"))
    if frost_files:
        chosen = frost_files[np.random.randint(len(frost_files))]
        loaded = cv2.imread(str(chosen))
        if loaded is not None:
            loaded = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
            frost_pattern = cv2.resize(loaded, (w, h)).astype(np.float32)

    if frost_pattern is None:
        frost_pattern = _generate_procedural_frost(h, w)

    corrupted = img_w * arr + frost_w * frost_pattern
    corrupted = np.clip(corrupted, 0, 255).astype(np.uint8)

    return Image.fromarray(corrupted) if is_pil else corrupted
