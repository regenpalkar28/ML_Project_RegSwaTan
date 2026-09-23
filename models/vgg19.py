"""
VGG19 Architecture for Traffic Sign Recognition.
Implements VGG19 (with and without Batch Normalization) adapted for traffic sign datasets
(e.g., GTSRB, BelgiumTSC, CTSD) using PyTorch and torchvision.
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import (
    vgg19,
    vgg19_bn,
    VGG19_Weights,
    VGG19_BN_Weights,
)


class VGG19SignClassifier(nn.Module):
    """
    VGG19 architecture customized for traffic sign classification.
    
    Supports:
    - Standard VGG19 or VGG19 with Batch Normalization (vgg19_bn, highly recommended)
    - Optional ImageNet pre-trained weights for transfer learning
    - Compact classification head optimized for small traffic sign images (32x32, 48x48)
      to drastically reduce parameters and CPU/GPU memory footprint, or standard head.
    """

    def __init__(
        self,
        num_classes: int = 43,
        pretrained: bool = False,
        use_batch_norm: bool = True,
        in_channels: int = 3,
        dropout_rate: float = 0.5,
        compact_head: bool = True,
    ):
        """
        Args:
            num_classes (int): Number of target classes (default: 43 for GTSRB).
            pretrained (bool): Whether to initialize backbone with ImageNet pre-trained weights.
            use_batch_norm (bool): If True, uses VGG19-BN (much faster & stabler convergence).
            in_channels (int): Input image channels (3 for RGB).
            dropout_rate (float): Dropout probability in the classification head.
            compact_head (bool): If True, adapts feature pooling and classifier head for
                                low-resolution images (reduces params from ~143M to ~21M,
                                providing significant speedup for CPU/GPU training).
        """
        super(VGG19SignClassifier, self).__init__()

        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        self.compact_head = compact_head

        # Select weights if pretrained
        if use_batch_norm:
            weights = VGG19_BN_Weights.DEFAULT if pretrained else None
            base_model = vgg19_bn(weights=weights)
        else:
            weights = VGG19_Weights.DEFAULT if pretrained else None
            base_model = vgg19(weights=weights)

        # Handle non-RGB input channels if requested
        if in_channels != 3:
            first_conv = base_model.features[0]
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=(first_conv.bias is not None),
            )
            base_model.features[0] = new_conv

        self.features = base_model.features

        if compact_head:
            # Pooling to (2, 2) feature map: 512 * 2 * 2 = 2048 features
            # Drastically cuts parameters from ~143M to ~21M
            self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 2 * 2, 512),
                nn.BatchNorm1d(512) if use_batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256) if use_batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(256, num_classes),
            )
        else:
            # Standard VGG head with 7x7 pooling (classic ~143M params)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(4096, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input batch of images (B, C, H, W)
        Returns:
            torch.Tensor: Unnormalized class logits (B, num_classes)
        """
        x = self.features(x)
        x = self.adaptive_pool(x)
        logits = self.classifier(x)
        return logits


def get_vgg19_model(
    num_classes: int = 43,
    pretrained: bool = False,
    use_batch_norm: bool = True,
    in_channels: int = 3,
    compact_head: bool = True,
) -> VGG19SignClassifier:
    """
    Factory function to instantiate the VGG19 model.

    Args:
        num_classes (int): Number of target classes.
        pretrained (bool): Load ImageNet pre-trained weights.
        use_batch_norm (bool): Use Batch Normalization variant (vgg19_bn).
        in_channels (int): Number of input channels (3 for RGB).
        compact_head (bool): Use lightweight classification head suitable for 32x32/48x48.

    Returns:
        VGG19SignClassifier: Instantiated VGG19 model.
    """
    return VGG19SignClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        use_batch_norm=use_batch_norm,
        in_channels=in_channels,
        compact_head=compact_head,
    )


if __name__ == "__main__":
    print("Testing VGG19SignClassifier initialization...")
    # Test compact head
    model_compact = get_vgg19_model(num_classes=43, compact_head=True, use_batch_norm=True)
    compact_params = sum(p.numel() for p in model_compact.parameters())
    print(f"VGG19-BN (Compact Head) Parameters: {compact_params:,} (~{compact_params * 4 / 1024 / 1024:.1f} MB)")

    # Test forward pass with 32x32 dummy input
    dummy_input = torch.randn(2, 3, 32, 32)
    out_compact = model_compact(dummy_input)
    print(f"Input shape: {dummy_input.shape} -> Compact Output shape: {out_compact.shape}")
    assert out_compact.shape == (2, 43), "Compact head shape mismatch"

    # Test standard head
    model_std = get_vgg19_model(num_classes=43, compact_head=False, use_batch_norm=True)
    std_params = sum(p.numel() for p in model_std.parameters())
    print(f"VGG19-BN (Standard Head) Parameters: {std_params:,} (~{std_params * 4 / 1024 / 1024:.1f} MB)")
    out_std = model_std(dummy_input)
    print(f"Input shape: {dummy_input.shape} -> Standard Output shape: {out_std.shape}")
    assert out_std.shape == (2, 43), "Standard head shape mismatch"

    print("All VGG19 tests PASSED successfully!")