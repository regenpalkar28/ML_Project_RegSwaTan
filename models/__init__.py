from .RawCNN import BasicSignClassifier, get_model

try:
    from .vgg19_model import VGG19SignClassifier, get_vgg19_model
    __all__ = ["BasicSignClassifier", "VGG19SignClassifier", "get_model", "get_vgg19_model"]
except (ImportError, AttributeError):
    __all__ = ["BasicSignClassifier", "get_model"]

