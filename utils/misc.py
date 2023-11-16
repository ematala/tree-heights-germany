import argparse
import os
from random import seed as pyseed
from typing import List, Optional, Tuple

from numpy import histogram as nphist
from numpy import ndarray, stack
from numpy.random import seed as npseed
from requests import post
from torch import device as Device
from torch import manual_seed as tseed
from torch.backends.mps import is_available as mps_available
from torch.cuda import is_available as cuda_available
from torch.cuda import manual_seed as cseed
from torch.cuda import manual_seed_all as cseed_all


def get_window_bounds(
    idx: int,
    patch_size: int,
    image_size: int = 4096,
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

    # Prevent division by zero
    epsilon = 1e-8

    # Calculate NDVI
    NDVI = (NIR - R) / (NIR + R + epsilon)

    # Normalize NDVI into range [0, 1]
    NDVI = (NDVI + 1) / 2

    # Normalize image channels into range [0, 1]
    R, G, B, NIR = normalize(img)

    # Ensemble image
    return stack([R, G, B, NIR, NDVI])


def get_label_bins(
    label: ndarray,
    bins: List[int] = list(range(0, 55, 5)),
    no_data: float = 0,
) -> ndarray:
    count, _ = nphist(label[label != no_data], bins)
    return count


def seed_everyting(seed: int = 42):
    pyseed(seed)
    npseed(seed)
    tseed(seed)
    cseed(seed)
    cseed_all(seed)


def get_device(dev: Optional[str] = None) -> Device:
    return Device(
        dev
        if dev
        else "cuda"
        if cuda_available()
        else "mps"
        if mps_available()
        else "cpu"
    )


def send_telegram_message(message: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise ValueError("Telegram token and chat id must be set")

    post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        {"chat_id": chat_id, "text": message},
    )


def get_training_args():
    """Get arguments from command line

    Returns:
        Namespace: Arguments
    """
    parser = argparse.ArgumentParser(
        description="Train a selected model on predicting tree canopy heights"
    )
    parser.add_argument(
        "--model",
        choices=[
            "unet",
            "unetplusplus",
            "vit",
            "vit-base",
            "vit-medium",
            "vit-large",
        ],
        default="vit-medium",
        help="Model type [default: vit-medium]",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Training epochs [default: 50]"
    )

    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size [default: 64]"
    )

    parser.add_argument(
        "--notify",
        type=bool,
        default=True,
        help="Notify after training [default: True]",
    )

    parser.add_argument(
        "--teacher",
        type=str,
        default=None,
        help="Teacher model [default: None]",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha for knowledge distillation [default: 0.5]",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience for early stopping [default: 10]",
    )

    return parser.parse_args()
