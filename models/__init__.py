from .unet import Unet
from .unetplusplus import UnetPlusPlus
from .vit import Vit
from .vitnet import VitNet
from .base import BaseModel

_models = {
    "unet": Unet(),
    "unetplusplus": UnetPlusPlus(),
    "vit": Vit(),
    "vit-base": VitNet(
        num_attention_heads=8,
        hidden_size=128,
        intermediate_size=512,
    ),
    "vit-medium": VitNet(
        num_attention_heads=12,
        hidden_size=192,
        intermediate_size=768,
    ),
    "vit-large": VitNet(
        num_attention_heads=16, hidden_size=256, intermediate_size=1024
    ),
}


def get_all_models():
    return list(_models.keys())


def make_model(name: str):
    if name not in _models:
        raise NotImplementedError(f"Model {name} not implemented")
    return _models[name]
