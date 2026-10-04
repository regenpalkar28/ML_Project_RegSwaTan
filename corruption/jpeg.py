"""
JPEG Compression Artifacts Corruption (Hendrycks & Dietterich benchmark).
Simulates DCT block loss and ringing artifacts from aggressive lossy compression.
"""

from io import BytesIO
from typing import Union
import numpy as np
from PIL import Image


def jpeg_compression(image: Union[Image.Image, np.ndarray], severity: int = 1) -> Union[Image.Image, np.ndarray]:
    """
    Applies lossy JPEG compression artifacts across 5 severity levels.

    Args:
        image: PIL Image or NumPy array (H, W, C) in range [0, 255].
        severity: Integer from 1 (mild compression) to 5 (extreme compression artifacts).

    Returns:
        Corrupted image in the same format as input.
    """
    if not (1 <= severity <= 5):
        raise ValueError(f"Severity must be in range [1, 5], got {severity}")

    is_pil = isinstance(image, Image.Image)
    pil_img = image if is_pil else Image.fromarray(np.array(image, dtype=np.uint8))

    # JPEG quality factor: lower quality = heavier compression artifacts
    qualities = [55, 38, 24, 14, 6]
    quality = qualities[severity - 1]

    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    # Load pixels into memory before closing buffer
    compressed.load()

    return compressed if is_pil else np.array(compressed)
