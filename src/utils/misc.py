from random import seed as pyseed
from typing import List, Tuple

from numpy import histogram as nphist
from numpy import ndarray
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


def normalize_image(img: ndarray) -> ndarray:
    return (img - img.min()) / (img.max() - img.min())


def get_label_bins(
    label: ndarray, bins: List[int] = list(range(0, 55, 5)), no_data: float = 0
) -> ndarray:
    count, _ = nphist(label[label != no_data], bins)
    return count


def seed(s: int = 42):
    pyseed(s)
    npseed(s)
    tseed(s)
    cseed(s)
    cseed_all(s)
