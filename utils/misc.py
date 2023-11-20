import os
from random import seed as pyseed
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from numpy import histogram as nphist
from numpy import ndarray
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


def get_num_processes_to_spawn(
    num_images: int,
    max_processes_per_core: int = 1,
) -> int:
    return min(os.cpu_count() * max_processes_per_core, num_images)


def send_telegram_message(message: str):
    load_dotenv()
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise ValueError("Telegram token and chat id must be set")

    post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        {"chat_id": chat_id, "text": message},
    )
