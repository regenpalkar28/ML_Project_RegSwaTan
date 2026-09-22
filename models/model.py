import torch
import torch.nn as nn


class BasicSignClassifier(nn.Module):
    """
    A robust baseline Convolutional Neural Network (CNN) designed for
    traffic sign recognition (e.g., GTSRB, BTSD, CTSD).
    
    Architecture:
    - 3 Convolutional blocks with Batch Normalization, ReLU, and Dropout
    - Adaptive Average Pooling to handle variable input dimensions (32x32, 48x48, etc.)
    - Fully connected classification head with Dropout regularization
    """

    def __init__(self, num_classes: int = 43, in_channels: int = 3, dropout_rate: float = 0.3):
        super(BasicSignClassifier, self).__init__()
        
        self.num_classes = num_classes

        # Feature Extractor
        # Block 1: Input (B, C, H, W) -> (B, 32, H/2, W/2)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate * 0.5)
        )

        # Block 2: (B, 32, H/2, W/2) -> (B, 64, H/4, W/4)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate * 0.75)
        )

        # Block 3: (B, 64, H/4, W/4) -> (B, 128, H/8, W/8)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate)
        )

        # Adaptive pooling ensures fixed feature map size (4x4) regardless of input resolution
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input batch of images (B, C, H, W)
        Returns:
            torch.Tensor: Unnormalized class logits (B, num_classes)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.adaptive_pool(x)
        logits = self.classifier(x)
        return logits


def get_model(
    num_classes: int = 43,
    model_type: str = "basic_cnn",
    in_channels: int = 3,
    **kwargs,
) -> nn.Module:
    """
    Factory function to instantiate the model.

    Args:
        num_classes (int): Number of target traffic sign classes (default: 43 for GTSRB).
        model_type (str): Type of model architecture ('basic_cnn', 'vgg19', 'vgg19_bn').
        in_channels (int): Number of input image channels (3 for RGB).
        **kwargs: Additional parameters forwarded to specific models (e.g., pretrained, compact_head).

    Returns:
        nn.Module: Instantiated PyTorch model.
    """
    mtype = model_type.lower()
    if mtype in ["basic_cnn", "cnn"]:
        return BasicSignClassifier(num_classes=num_classes, in_channels=in_channels, **kwargs)
    elif mtype in ["vgg19", "vgg19_bn"]:
        from .vgg19 import get_vgg19_model
        use_bn = kwargs.pop("use_batch_norm", (mtype == "vgg19_bn"))
        return get_vgg19_model(
            num_classes=num_classes,
            use_batch_norm=use_bn,
            in_channels=in_channels,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Supported types: ['basic_cnn', 'vgg19', 'vgg19_bn']"
        )


if __name__ == "__main__":
    # Quick self-test verifying shape compatibility
    test_model = get_model(num_classes=43)
    dummy_input = torch.randn(4, 3, 32, 32)
    output = test_model(dummy_input)
    print(f"Model successfully built.")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Expected: [4, 43])")
