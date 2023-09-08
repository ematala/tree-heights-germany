from typing import List, Tuple

from numpy import histogram as nphist
from numpy import ndarray


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


def get_bins(
    label: ndarray, bins: List[int] = list(range(0, 55, 5)), no_data: float = 0
) -> ndarray:
    count, _ = nphist(label[label != no_data], bins)
    return count
