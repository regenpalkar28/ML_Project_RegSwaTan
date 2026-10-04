"""
Benchmark Image Corruptions for Robustness Evaluation.
Implements the 14 standard Hendrycks & Dietterich corruptions across 5 severity levels
for Traffic Sign Recognition (GTSRB, BelgiumTSC, CTSD).
"""

from typing import Callable, Dict, List, Union
import numpy as np
from PIL import Image
import torch

from .brightness import brightness
from .contrast import contrast
from .defocus_blur import defocus_blur
from .elastic import elastic_transform
from .fog import fog
from .frost import frost
from .gaussian import gaussian_noise
from .impulse import impulse_noise
from .jpeg import jpeg_compression
from .motion_blur import motion_blur
from .pixelate import pixelate
from .shot import shot_noise
from .snow import snow
from .zoom_blur import zoom_blur


# Canonical dictionary mapping corruption names and aliases to their callables
CORRUPTIONS_DICT: Dict[str, Callable] = {
    # Noise corruptions
    "gaussian_noise": gaussian_noise,
    "gaussian": gaussian_noise,
    "shot_noise": shot_noise,
    "shot": shot_noise,
    "impulse_noise": impulse_noise,
    "impulse": impulse_noise,
    
    # Blur corruptions
    "defocus_blur": defocus_blur,
    "defocus": defocus_blur,
    "motion_blur": motion_blur,
    "motion": motion_blur,
    "zoom_blur": zoom_blur,
    "zoom": zoom_blur,
    
    # Weather corruptions
    "fog": fog,
    "frost": frost,
    "snow": snow,
    
    # Digital / Photometric corruptions
    "brightness": brightness,
    "contrast": contrast,
    "pixelate": pixelate,
    "elastic_transform": elastic_transform,
    "elastic": elastic_transform,
    "jpeg_compression": jpeg_compression,
    "jpeg": jpeg_compression,
}

# The 14 canonical benchmark corruption names
BENCHMARK_CORRUPTIONS: List[str] = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "motion_blur",
    "zoom_blur",
    "fog",
    "frost",
    "snow",
    "brightness",
    "contrast",
    "pixelate",
    "elastic_transform",
    "jpeg_compression",
]

# Categorized dictionary matching research groupings
CORRUPTION_CATEGORIES: Dict[str, List[str]] = {
    "Noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "Blur": ["defocus_blur", "motion_blur", "zoom_blur"],
    "Weather": ["fog", "frost", "snow"],
    "Digital": ["brightness", "contrast", "pixelate", "elastic_transform", "jpeg_compression"],
}


def apply_corruption(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    corruption_type: str,
    severity: int = 1,
) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """
    Apply a specified corruption at a given severity level.

    Args:
        image: PIL Image, NumPy array (H, W, C) in [0, 255], or PyTorch Tensor (C, H, W) in [0, 1] or [0, 255].
        corruption_type: Name of the corruption (e.g. 'gaussian_noise', 'fog', 'motion_blur').
        severity: Severity level integer from 1 to 5.

    Returns:
        Corrupted image in the matching input type.
    """
    key = corruption_type.lower().strip()
    if key not in CORRUPTIONS_DICT:
        raise ValueError(
            f"Unknown corruption type '{corruption_type}'. Supported types: {BENCHMARK_CORRUPTIONS}"
        )

    fn = CORRUPTIONS_DICT[key]

    # Handle PyTorch Tensor input
    if isinstance(image, torch.Tensor):
        is_cuda = image.is_cuda
        dev = image.device
        orig_dtype = image.dtype

        # Detach and convert (C, H, W) tensor to NumPy (H, W, C)
        tensor_np = image.detach().cpu().numpy()
        if tensor_np.ndim == 3 and tensor_np.shape[0] in [1, 3]:
            tensor_np = np.transpose(tensor_np, (1, 2, 0))

        # Check if tensor was normalized [0, 1]
        was_float_01 = tensor_np.max() <= 1.0 and np.issubdtype(tensor_np.dtype, np.floating)
        if was_float_01:
            tensor_np = (tensor_np * 255.0).astype(np.uint8)
        else:
            tensor_np = tensor_np.astype(np.uint8)

        # Apply corruption
        out_np = fn(tensor_np, severity=severity)
        if isinstance(out_np, Image.Image):
            out_np = np.array(out_np)

        if was_float_01:
            out_np = out_np.astype(np.float32) / 255.0

        # Convert back to (C, H, W) Tensor
        if out_np.ndim == 3:
            out_np = np.transpose(out_np, (2, 0, 1))

        out_tensor = torch.from_numpy(out_np).to(device=dev, dtype=orig_dtype)
        return out_tensor

    return fn(image, severity=severity)


__all__ = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "motion_blur",
    "zoom_blur",
    "fog",
    "frost",
    "snow",
    "brightness",
    "contrast",
    "pixelate",
    "elastic_transform",
    "jpeg_compression",
    "CORRUPTIONS_DICT",
    "BENCHMARK_CORRUPTIONS",
    "CORRUPTION_CATEGORIES",
    "apply_corruption",
]
