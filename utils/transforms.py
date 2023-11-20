import random

import numpy as np
import torch


def to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()


def normalize(tensor: torch.Tensor, max_value=65535.0) -> torch.Tensor:
    # Normalize only the first four channels (R, G, B, NIR)
    tensor[:4] = tensor[:4] / max_value

    return tensor


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
