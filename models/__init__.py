from .unet import Unet
from .unetplusplus import UnetPlusPlus
from .vit import Vit
from .base import BaseModel

_models = {
    "unet": Unet(),
    "unetplusplus": UnetPlusPlus(),
    "vit-tiny": Vit(backbone="vit_tiny_patch16_256", embed_dim=192),
    "vit-small": Vit(backbone="vit_small_patch16_256", embed_dim=384),
    "vit-base": Vit(backbone="vit_base_patch16_256", embed_dim=768),
    "vit-large": Vit(backbone="vit_large_patch16_256", embed_dim=1024),
}


def get_all_models():
    return list(_models.keys())


def make_model(name: str) -> BaseModel:
    if name not in _models:
        raise NotImplementedError(f"Model {name} not implemented")
    return _models[name]
