from .RawCNN import BasicSignClassifier, get_model

try:
    from .vgg19_model import (
        VGG19SignClassifier,
        VGG19BN_MSFF,
        VGG19BN_MSFF_Attn,
        get_vgg19_model,
        get_vgg19_msff_model,
        get_vgg19_msff_attn_model,
    )
    __all__ = [
        "BasicSignClassifier",
        "VGG19SignClassifier",
        "VGG19BN_MSFF",
        "VGG19BN_MSFF_Attn",
        "get_model",
        "get_vgg19_model",
        "get_vgg19_msff_model",
        "get_vgg19_msff_attn_model",
    ]
except (ImportError, AttributeError):
    __all__ = ["BasicSignClassifier", "get_model"]


