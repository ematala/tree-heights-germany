from .unet import Unet
from .unetplusplus import UnetPlusPlus
from .vit import Vit
from .vitnet import VitNet
from .base import BaseModel


def make_model(name: str = "vit-medium"):
    return {
        "vit-medium": Vit(features=128),
        # "large": Vit(features=256, backbone="vitl16_384"),
        # "deit": Vit(features=128, backbone="vit_deit_base_patch16_384"),
        # "hybrid": Vit(features=128, backbone="vitb_rn50_384"),
    }[name]
