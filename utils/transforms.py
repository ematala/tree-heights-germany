import random

import numpy as np
import torch

_mean = torch.tensor([550.9839, 600.3104, 335.5715, 3093.5127])
_std = torch.tensor([417.7791, 287.0238, 217.3390, 891.1005])


def to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32)).float()


def normalize(
    tensor: torch.Tensor,
    mean: torch.Tensor = _mean,
    std: torch.Tensor = _std,
) -> torch.Tensor:
    # Normalize only the first four channels (R, G, B, NIR)
    tensor[:4] = (tensor[:4] - mean.view(-1, 1, 1)[:4]) / std.view(-1, 1, 1)[:4]

    return tensor


def denormalize(
    tensor: torch.Tensor,
    mean: torch.Tensor = _mean,
    std: torch.Tensor = _std,
) -> torch.Tensor:
    # Deormalize only the first four channels (R, G, B, NIR)
    tensor[:4] = tensor[:4] * std.view(-1, 1, 1)[:4] + mean.view(-1, 1, 1)[:4]

    return tensor


def scale(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def random_vertical_flip(image: torch.Tensor, label: torch.Tensor, prob=0.5):
    if random.random() < prob:
        return image.flip(-2), label.flip(-2)
    return image, label


def random_horizontal_flip(image: torch.Tensor, label: torch.Tensor, prob=0.5):
    if random.random() < prob:
        return image.flip(-1), label.flip(-1)
    return image, label


def random_rotation(image: torch.Tensor, label: torch.Tensor, prob=0.5):
    if random.random() < prob:
        angles = [0, 90, 180, 270]
        angle = random.choice(angles)
        rotations = angle // 90
        return torch.rot90(image, rotations, [-2, -1]), torch.rot90(
            label, rotations, [-2, -1]
        )
    return image, label


def add_ndvi(image: np.ndarray) -> np.ndarray:
    # unpack image
    R, G, B, NIR = image.astype(np.float32)

    # prevent division by zero
    epsilon = 1e-8

    # calculate NDVI
    NDVI = (NIR - R) / (NIR + R + epsilon)

    # normalize NDVI to the range [0, 1]
    NDVI = (NDVI + 1) / 2

    return np.stack([R, G, B, NIR, NDVI])
