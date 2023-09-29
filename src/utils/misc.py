from random import seed as pyseed
from typing import List, Tuple

from numpy import histogram as nphist
from numpy import ndarray, stack
from numpy.random import seed as npseed
from torch import manual_seed as tseed
from torch.cuda import manual_seed as cseed
from torch.cuda import manual_seed_all as cseed_all


def get_window_bounds(
    idx: int, patch_size: int, image_size: int = 4096
) -> Tuple[int, int, int, int]:
    row = (idx // (image_size // patch_size)) * patch_size
    col = (idx % (image_size // patch_size)) * patch_size

    return (
        max(0, row),
        min(image_size, row + patch_size),
        max(0, col),
        min(image_size, col + patch_size),
    )


def normalize(arr: ndarray) -> ndarray:
    return (arr - arr.min()) / (arr.max() - arr.min())


def get_normalized_image(src) -> ndarray:
    # Swap order of bands
    img = src.read([3, 2, 1, 4])

    # Unpack image
    R, G, B, NIR = img

    # Calculate NDVI
    NDVI = (NIR - R) / (NIR + R)

    # Normalize NDVI into range [0, 1]
    NDVI = (NDVI + 1) / 2

    # Normalize image channels into range [0, 1]
    R, G, B, NIR = normalize(img)

    # Ensemble image
    return stack([R, G, B, NIR, NDVI])


def get_label_bins(
    label: ndarray, bins: List[int] = list(range(0, 55, 5)), no_data: float = 0
) -> ndarray:
    count, _ = nphist(label[label != no_data], bins)
    return count


def seed_everyting(seed: int = 42):
    pyseed(seed)
    npseed(seed)
    tseed(seed)
    cseed(seed)
    cseed_all(seed)
