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
        **kwargs,
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
    **kwargs,
) -> VGG19SignClassifier:
    """
    Factory function to instantiate the VGG19 model.

    Args:
        num_classes (int): Number of target classes.
        pretrained (bool): Load ImageNet pre-trained weights.
        use_batch_norm (bool): Use Batch Normalization variant (vgg19_bn).
        in_channels (int): Input channels (3 for RGB).
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


class VGG19BN_MSFF(nn.Module):
    """
    VGG19-BN with Multi-Scale Feature Fusion (MSFF) without SE Attention (for ablation studies).
    
    Architecture:
    - Backbone: Sliced VGG19-BN stages (Block 1-2, Block 3, Block 4, Block 5)
    - Intermediate Feature Extraction:
        - Block 3 (after maxpool 3, 256 channels): 1x1 Conv + BN + ReLU projection to 128 channels,
          then downsampled via AdaptiveAvgPool2d((2, 2)) -> (B, 128, 2, 2)
        - Block 4 (after maxpool 4, 512 channels): 1x1 Conv + BN + ReLU projection to 256 channels,
          then downsampled via AdaptiveAvgPool2d((2, 2)) -> (B, 256, 2, 2)
        - Block 5 (after maxpool 5, 512 channels): AdaptiveAvgPool2d((2, 2)) -> (B, 512, 2, 2)
    - Feature Concatenation:
        - Flattened features concatenated into a unified multi-scale vector:
          (128 + 256 + 512) * 2 * 2 = 3584 dimensions.
    - Classification Head: Linear layer (3584 -> num_classes) with Dropout.
    """

    def __init__(
        self,
        num_classes: int = 43,
        pretrained: bool = False,
        in_channels: int = 3,
        dropout_rate: float = 0.5,
        use_se: bool = False,
        se_reduction: int = 16,
        **kwargs,
    ):
        """
        Args:
            num_classes (int): Number of target classes (default 43 for GTSRB).
            pretrained (bool): Whether to initialize backbone with ImageNet pre-trained weights.
            in_channels (int): Number of input channels (default 3 for RGB).
            dropout_rate (float): Dropout probability before classification head.
            use_se (bool): Whether to apply Squeeze-and-Excitation channel attention.
            se_reduction (int): Reduction ratio r for SE attention (default 16).
        """
        super(VGG19BN_MSFF, self).__init__()

        self.num_classes = num_classes
        self.use_se = use_se
        self.se_reduction = se_reduction

        weights = VGG19_BN_Weights.DEFAULT if pretrained else None
        base_model = vgg19_bn(weights=weights)

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

        # Sliced VGG19-BN feature extraction stages:
        # Block 1 & 2: layers 0..13 (ends with MaxPool 2, 128 channels)
        self.block1_2 = base_model.features[0:14]
        # Block 3: layers 14..26 (ends with MaxPool 3, 256 channels)
        self.block3 = base_model.features[14:27]
        # Block 4: layers 27..39 (ends with MaxPool 4, 512 channels)
        self.block4 = base_model.features[27:40]
        # Block 5: layers 40..52 (ends with MaxPool 5, 512 channels)
        self.block5 = base_model.features[40:53]

        # Spatial alignment pool to (2, 2)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        # 1x1 Convolution + BatchNorm + ReLU projection layers:
        # Block 3 projection: 256 -> 128 channels
        self.proj3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Block 4 projection: 512 -> 256 channels
        self.proj4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Total multi-scale fused dimension: (128 + 256 + 512) * 2 * 2 = 3584
        self.fused_dim = (128 + 256 + 512) * 2 * 2

        # Squeeze-and-Excitation (SE) Channel Attention block
        if self.use_se:
            reduced_dim = max(1, self.fused_dim // se_reduction)  # 3584 // 16 = 224
            self.se = nn.Sequential(
                nn.Linear(self.fused_dim, reduced_dim, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(reduced_dim, self.fused_dim, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.se = None

        # Final Linear classification head for GTSRB
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.fused_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale feature fusion and optional SE attention.
        Args:
            x (torch.Tensor): Input batch of images (B, C, H, W)
        Returns:
            torch.Tensor: Unnormalized class logits (B, num_classes)
        """
        # Backbone intermediate feature extraction
        x12 = self.block1_2(x)
        b3_out = self.block3(x12)       # (B, 256, H/4, W/4)
        b4_out = self.block4(b3_out)    # (B, 512, H/8, W/8)
        b5_out = self.block5(b4_out)    # (B, 512, H/16, W/16)

        # Downsample and project multi-scale features to (2, 2)
        p3 = self.pool(self.proj3(b3_out)).flatten(1)  # (B, 128 * 2 * 2 = 512)
        p4 = self.pool(self.proj4(b4_out)).flatten(1)  # (B, 256 * 2 * 2 = 1024)
        p5 = self.pool(b5_out).flatten(1)              # (B, 512 * 2 * 2 = 2048)

        # Concatenate into unified multi-scale feature vector (B, 3584)
        fused = torch.cat([p3, p4, p5], dim=1)

        # Dynamic channel re-weighting via SE Attention
        if self.se is not None:
            attn_weights = self.se(fused)
            fused = fused * attn_weights

        logits = self.classifier(fused)
        return logits


class VGG19BN_MSFF_Attn(VGG19BN_MSFF):
    """
    VGG19-BN with Multi-Scale Feature Fusion (MSFF) and Squeeze-and-Excitation (SE) Channel Attention.
    """

    def __init__(
        self,
        num_classes: int = 43,
        pretrained: bool = False,
        in_channels: int = 3,
        dropout_rate: float = 0.5,
        se_reduction: int = 16,
        **kwargs,
    ):
        super(VGG19BN_MSFF_Attn, self).__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            in_channels=in_channels,
            dropout_rate=dropout_rate,
            use_se=True,
            se_reduction=se_reduction,
        )


def get_vgg19_msff_model(
    num_classes: int = 43,
    pretrained: bool = False,
    in_channels: int = 3,
    dropout_rate: float = 0.5,
    use_se: bool = False,
    se_reduction: int = 16,
    **kwargs,
) -> VGG19BN_MSFF:
    """Factory function for VGG19-BN MSFF model (without SE attention)."""
    return VGG19BN_MSFF(
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        dropout_rate=dropout_rate,
        use_se=use_se,
        se_reduction=se_reduction,
    )


def get_vgg19_msff_attn_model(
    num_classes: int = 43,
    pretrained: bool = False,
    in_channels: int = 3,
    dropout_rate: float = 0.5,
    se_reduction: int = 16,
    **kwargs,
) -> VGG19BN_MSFF_Attn:
    """Factory function for VGG19-BN MSFF model with Squeeze-and-Excitation (SE) Attention."""
    return VGG19BN_MSFF_Attn(
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        dropout_rate=dropout_rate,
        se_reduction=se_reduction,
    )


if __name__ == "__main__":
    print("Testing VGG19 Models...")
    dummy_input = torch.randn(2, 3, 32, 32)

    # 1. Test Baseline VGG19-BN (Compact Head)
    model_compact = get_vgg19_model(num_classes=43, compact_head=True, use_batch_norm=True)
    compact_params = sum(p.numel() for p in model_compact.parameters())
    out_compact = model_compact(dummy_input)
    print(f"1. VGG19-BN (Compact Head): Parameters={compact_params:,} | Output shape={out_compact.shape}")
    assert out_compact.shape == (2, 43)

    # 2. Test VGG19-BN MSFF (Ablation: No SE Attention)
    model_msff = get_vgg19_msff_model(num_classes=43)
    msff_params = sum(p.numel() for p in model_msff.parameters())
    out_msff = model_msff(dummy_input)
    print(f"2. VGG19-BN MSFF:           Parameters={msff_params:,} | Output shape={out_msff.shape}")
    assert out_msff.shape == (2, 43)

    # 3. Test VGG19-BN MSFF + SE Attention
    model_msff_attn = get_vgg19_msff_attn_model(num_classes=43)
    msff_attn_params = sum(p.numel() for p in model_msff_attn.parameters())
    out_msff_attn = model_msff_attn(dummy_input)
    print(f"3. VGG19-BN MSFF + Attn:    Parameters={msff_attn_params:,} | Output shape={out_msff_attn.shape}")
    assert out_msff_attn.shape == (2, 43)

    print("All VGG19, MSFF, and MSFF-Attn models verified successfully!")
