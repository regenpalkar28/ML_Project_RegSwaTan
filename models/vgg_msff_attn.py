"""
Multi-Scale Feature Fusion (MSFF) and Squeeze-and-Excitation (SE) Channel Attention
for VGG19-BN Traffic Sign Recognition (GTSRB).

Architecture Overview:
----------------------
1. Feature Extraction:
   - Block 1 & 2: features[0:14]
   - Block 3: features[14:27] -> 256 channels (after MaxPool 3)
   - Block 4: features[27:40] -> 512 channels (after MaxPool 4)
   - Block 5: features[40:53] -> 512 channels (after MaxPool 5)

2. Multi-Scale Alignment & Projection:
   - Block 3: AdaptiveAvgPool2d((2, 2)) + 1x1 Conv(256 -> 128) + BatchNorm2d + ReLU
   - Block 4: AdaptiveAvgPool2d((2, 2)) + 1x1 Conv(512 -> 256) + BatchNorm2d + ReLU
   - Block 5: AdaptiveAvgPool2d((2, 2)) -> 512 channels

3. Fusion:
   - Flatten & concatenate: (128 + 256 + 512) * 2 * 2 = 3584 dims.

4. SE Channel Attention (VGG19BN_MSFF_Attn):
   - Reduction ratio r = 16: Linear(3584 -> 224) -> ReLU -> Linear(224 -> 3584) -> Sigmoid
   - Feature re-weighting: z_gated = z * attention_weights

5. Classification Head:
   - Dropout(p=0.5) + Linear(3584 -> num_classes)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision.models import VGG19_BN_Weights, vgg19_bn


class VGG19BN_MSFF_Attn(nn.Module):
    """
    VGG19-BN with Multi-Scale Feature Fusion (MSFF) and Squeeze-and-Excitation (SE)
    Channel Attention for Traffic Sign Recognition.

    Extracts multi-scale representations from intermediate stages (Blocks 3, 4, 5),
    projects them to balanced channel capacities, pools to a uniform (2, 2) spatial
    grid, flattens to 3584 dims, and applies SE channel gating.
    """

    def __init__(
        self,
        num_classes: int = 43,
        pretrained: bool = False,
        in_channels: int = 3,
        dropout_rate: float = 0.5,
        se_reduction: int = 16,
        use_se: bool = True,
        backbone: Optional[nn.Module] = None,
    ):
        """
        Args:
            num_classes (int): Number of target traffic sign classes (default: 43 for GTSRB).
            pretrained (bool): Load ImageNet pre-trained weights for VGG19-BN backbone.
            in_channels (int): Input image channels (default: 3 for RGB).
            dropout_rate (float): Dropout probability before final linear classifier.
            se_reduction (int): Reduction ratio 'r' in the Squeeze-and-Excitation block.
            use_se (bool): If True, applies SE channel attention. If False, operates as MSFF ablation.
            backbone (Optional[nn.Module]): Pre-instantiated VGG19-BN backbone.
        """
        super().__init__()
        self.num_classes = num_classes
        self.use_se = use_se
        self.se_reduction = se_reduction

        # 1. Initialize or accept VGG19-BN backbone
        if backbone is not None:
            vgg = backbone
        elif pretrained:
            weights = VGG19_BN_Weights.DEFAULT
            vgg = vgg19_bn(weights=weights)
        else:
            vgg = vgg19_bn(weights=None)

        # Support non-RGB input channels if requested
        if in_channels != 3:
            first_conv = vgg.features[0]
            vgg.features[0] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=(first_conv.bias is not None),
            )

        # Slice VGG19-BN into sequential blocks matching maxpool layer boundaries
        # features[0:14]  -> Blocks 1 & 2 (Conv + BN + ReLU x4 + MaxPool x2)
        # features[14:27] -> Block 3 (Conv + BN + ReLU x4 + MaxPool) -> 256 channels
        # features[27:40] -> Block 4 (Conv + BN + ReLU x4 + MaxPool) -> 512 channels
        # features[40:53] -> Block 5 (Conv + BN + ReLU x4 + MaxPool) -> 512 channels
        self.stage1_2 = nn.Sequential(*vgg.features[0:14])
        self.stage3 = nn.Sequential(*vgg.features[14:27])
        self.stage4 = nn.Sequential(*vgg.features[27:40])
        self.stage5 = nn.Sequential(*vgg.features[40:53])

        # 2. Spatial alignment to (2, 2)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        # 3. 1x1 Conv + BatchNorm + ReLU projection layers to balance channel dimensions
        # Block 3: 256 -> 128 channels
        self.proj_b3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Block 4: 512 -> 256 channels
        self.proj_b4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Block 5: keeps 512 channels

        # 4. Total fused multi-scale dimension:
        # (128 + 256 + 512) * 2 * 2 = 896 * 4 = 3584
        self.fused_dim = (128 + 256 + 512) * 2 * 2

        # 5. Squeeze-and-Excitation (SE) Channel Attention block
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

        # 6. Final Linear classification head for GTSRB
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.fused_dim, num_classes),
        )

    def extract_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pass through backbone and extract intermediate feature maps after
        MaxPool 3, MaxPool 4, and MaxPool 5.
        """
        x = self.stage1_2(x)
        f_b3 = self.stage3(x)   # (B, 256, H/4, W/4)
        f_b4 = self.stage4(f_b3) # (B, 512, H/8, W/8)
        f_b5 = self.stage5(f_b4) # (B, 512, H/16, W/16)
        return f_b3, f_b4, f_b5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale feature fusion and optional SE attention.
        Args:
            x (torch.Tensor): Input batch of images (B, C, H, W).
        Returns:
            torch.Tensor: Unnormalized class logits (B, num_classes).
        """
        # Extract Block 3, 4, 5 feature maps
        f_b3, f_b4, f_b5 = self.extract_features(x)

        # Spatially pool each block to (2, 2)
        p_b3 = self.pool(f_b3)
        p_b4 = self.pool(f_b4)
        p_b5 = self.pool(f_b5)

        # Apply 1x1 projection
        p_b3 = self.proj_b3(p_b3)  # (B, 128, 2, 2)
        p_b4 = self.proj_b4(p_b4)  # (B, 256, 2, 2)
        # Block 5 remains (B, 512, 2, 2)

        # Flatten features
        flat_b3 = torch.flatten(p_b3, 1)  # (B, 128 * 4 = 512)
        flat_b4 = torch.flatten(p_b4, 1)  # (B, 256 * 4 = 1024)
        flat_b5 = torch.flatten(p_b5, 1)  # (B, 512 * 4 = 2048)

        # Concatenate into unified multi-scale vector (B, 3584)
        fused = torch.cat([flat_b3, flat_b4, flat_b5], dim=1)

        # Apply SE channel attention if enabled
        if self.se is not None:
            weights = self.se(fused)
            fused = fused * weights

        # Final classification
        logits = self.classifier(fused)
        return logits


class VGG19BN_MSFF(VGG19BN_MSFF_Attn):
    """
    Ablation variant: VGG19-BN with Multi-Scale Feature Fusion (MSFF)
    WITHOUT Squeeze-and-Excitation Attention.
    """

    def __init__(
        self,
        num_classes: int = 43,
        pretrained: bool = False,
        in_channels: int = 3,
        dropout_rate: float = 0.5,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            in_channels=in_channels,
            dropout_rate=dropout_rate,
            se_reduction=16,
            use_se=False,
            backbone=backbone,
        )


def get_vgg19_msff_attn_model(
    num_classes: int = 43,
    pretrained: bool = False,
    in_channels: int = 3,
    dropout_rate: float = 0.5,
    se_reduction: int = 16,
) -> VGG19BN_MSFF_Attn:
    """Helper factory for VGG19-BN with MSFF and SE Channel Attention."""
    return VGG19BN_MSFF_Attn(
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        dropout_rate=dropout_rate,
        se_reduction=se_reduction,
        use_se=True,
    )


def get_vgg19_msff_model(
    num_classes: int = 43,
    pretrained: bool = False,
    in_channels: int = 3,
    dropout_rate: float = 0.5,
) -> VGG19BN_MSFF:
    """Helper factory for VGG19-BN with MSFF (without attention ablation)."""
    return VGG19BN_MSFF(
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        dropout_rate=dropout_rate,
    )


if __name__ == "__main__":
    print("Testing models/vgg_msff_attn.py models...")
    dummy_input = torch.randn(2, 3, 32, 32)

    # 1. MSFF + SE Attention
    model_attn = get_vgg19_msff_attn_model(num_classes=43)
    p_attn = sum(p.numel() for p in model_attn.parameters())
    out_attn = model_attn(dummy_input)
    print(f"VGG19BN_MSFF_Attn parameters: {p_attn:,} | Output shape: {out_attn.shape}")
    assert out_attn.shape == (2, 43), f"Expected (2, 43), got {out_attn.shape}"

    # 2. MSFF Ablation (No Attention)
    model_msff = get_vgg19_msff_model(num_classes=43)
    p_msff = sum(p.numel() for p in model_msff.parameters())
    out_msff = model_msff(dummy_input)
    print(f"VGG19BN_MSFF parameters:      {p_msff:,} | Output shape: {out_msff.shape}")
    assert out_msff.shape == (2, 43), f"Expected (2, 43), got {out_msff.shape}"

    print("Verification completed successfully!")
