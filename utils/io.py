from torch import device as Device
from torch import (
    load,
    save,
)
from torch.nn import Module


def load_model(path: str, device: Device) -> Module:
    model = load(path, map_location=device)

    model.eval()

    return model


def save_model(model: Module, path: str) -> None:
    save(model, path)
