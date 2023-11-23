from .unet import Unet
from .unetplusplus import UnetPlusPlus
from .vit import Vit
from .base import BaseModel

_models = {
    "unet": Unet(),
    "unetplusplus": UnetPlusPlus(),
    "vit-mini": Vit(embed_dim=86, num_heads=2),
    "vit-tiny": Vit(embed_dim=128, num_heads=4),
    "vit-small": Vit(embed_dim=192, num_heads=6),
    "vit-base": Vit(embed_dim=256, num_heads=8),
}


def get_all_models():
    return list(_models.keys())


def make_model(name: str) -> BaseModel:
    if name not in _models:
        raise NotImplementedError(f"Model {name} not implemented")
    return _models[name]
