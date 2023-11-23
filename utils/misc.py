import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from requests import post


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
    label: np.ndarray,
    bins: List[int] = list(range(0, 55, 5)),
    no_data: float = 0,
) -> np.ndarray:
    count, _ = np.histogram(label[label != no_data], bins)
    return count


def seed_everyting(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(dev: Optional[str] = None) -> torch.device:
    return torch.device(
        dev
        if dev
        else "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def get_num_processes_to_spawn(
    num_samples: int = np.Inf,
    max_processes_per_core: int = 1,
) -> int:
    return min((os.cpu_count() * max_processes_per_core) // 4, num_samples)


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
