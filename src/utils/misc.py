from typing import Tuple


def get_window_bounds(
    idx: int, patch_size: int, image_size: int = 4096
) -> Tuple[int, int, int, int]:
    row = (idx // (image_size // patch_size)) * patch_size
    col = (idx % (image_size // patch_size)) * patch_size
    return row, row + patch_size, col, col + patch_size
