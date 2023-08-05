from typing import Tuple


def get_window_bounds(
    idx: int, image_size: int = 4096, patch_size: int = 64
) -> Tuple[int, int]:
    row = (idx // (image_size // patch_size)) * patch_size
    col = (idx % (image_size // patch_size)) * patch_size
    return row, col
