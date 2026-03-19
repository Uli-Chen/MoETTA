CONFIG = dict()

from .config import Config
from .subconfigs import cifar10, cifar100, convnext, potpourri, vit_large

__all__ = ["Config", "CONFIG"]
