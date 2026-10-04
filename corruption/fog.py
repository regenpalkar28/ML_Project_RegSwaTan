"""
Fog Corruption (Hendrycks & Dietterich benchmark).
Simulates atmospheric haze and scattering using diamond-square plasma fractals.
"""

from typing import Union
import cv2
import numpy as np
from PIL import Image


def _plasma_fractal(mapsize: int = 64, wibbledecay: float = 3.0) -> np.ndarray:
    """
    Generate procedural cloud / fog heightmap using the diamond-square algorithm.
    Returns normalized float map in range [0, 1].
    """
    # Ensure mapsize is a power of 2
    mapsize = 1 << (mapsize - 1).bit_length()
    maparray = np.zeros((mapsize, mapsize), dtype=np.float32)
    stepsize = mapsize
    wibble = 100.0

    def wibbledmean(arr):
        return arr / 4.0 + wibble * np.random.uniform(-1.0, 1.0, arr.shape)

    while stepsize >= 2:
        # Square step
        half = stepsize // 2
        for y in range(half, mapsize, stepsize):
            for x in range(half, mapsize, stepsize):
                corners = [
                    maparray[y - half, x - half],
                    maparray[y - half, (x + half) % mapsize],
                    maparray[(y + half) % mapsize, x - half],
                    maparray[(y + half) % mapsize, (x + half) % mapsize],
                ]
                maparray[y, x] = float(np.mean(corners)) + wibble * float(np.random.uniform(-1.0, 1.0))

        # Diamond step
        for y in range(0, mapsize, half):
            for x in range((y + half) % stepsize, mapsize, stepsize):
                adjacent = [
                    maparray[(y - half) % mapsize, x],
                    maparray[(y + half) % mapsize, x],
                    maparray[y, (x - half) % mapsize],
                    maparray[y, (x + half) % mapsize],
                ]
                maparray[y, x] = float(np.mean(adjacent)) + wibble * float(np.random.uniform(-1.0, 1.0))

        stepsize //= 2
        wibble /= wibbledecay

    min_val, max_val = maparray.min(), maparray.max()
    if max_val > min_val:
        maparray = (maparray - min_val) / (max_val - min_val)
    return maparray


def fog(image: Union[Image.Image, np.ndarray], severity: int = 1) -> Union[Image.Image, np.ndarray]:
    """
    Applies atmospheric fog / haze corruption.

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

    # Fog density and decay parameters (Hendrycks benchmark)
    params = [(0.4, 2.8), (0.7, 2.5), (1.0, 2.0), (1.4, 1.8), (1.8, 1.5)]
    intensity, decay = params[severity - 1]

    # Generate fractal texture and resize to match image dimensions
    fractal = _plasma_fractal(mapsize=max(64, max(h, w)), wibbledecay=decay)
    fog_layer = cv2.resize(fractal, (w, h), interpolation=cv2.INTER_LINEAR)
    if arr.ndim == 3:
        fog_layer = fog_layer[..., np.newaxis]

    max_val = np.max(arr)
    arr = arr + intensity * fog_layer
    # Attenuate and re-normalize
    corrupted = np.clip(arr * max_val / (max_val + intensity), 0.0, 1.0) * 255.0
    corrupted = corrupted.astype(np.uint8)

    return Image.fromarray(corrupted) if is_pil else corrupted
