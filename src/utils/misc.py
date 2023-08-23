from typing import Tuple

from numpy import ndarray


def get_window_bounds(
    idx: int, patch_size: int, image_size: int = 4096, overlap: int = 0
) -> Tuple[int, int, int, int]:
    row = (idx // (image_size // patch_size)) * patch_size - overlap
    col = (idx % (image_size // patch_size)) * patch_size - overlap
    return (
        max(0, row),
        min(image_size, row + patch_size + 2 * overlap),
        max(0, col),
        min(image_size, col + patch_size + 2 * overlap),
    )


def normalize_image(img: ndarray) -> ndarray:
    return (img - img.min()) / (img.max() - img.min())
