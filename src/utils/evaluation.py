import logging
import os
from argparse import ArgumentParser
from logging import info
from typing import Tuple

from torch import Tensor, cat, no_grad
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import (
    get_data,
    get_device,
    load,
    loss,
    seed_everyting,
    test,
)
from .loss import filter


def get_args():
    """Get arguments from command line

    Returns:
        Namespace: Arguments
    """
    parser = ArgumentParser(
        description="evaluate a selected model on predicting tree canopy heights"
    )
    parser.add_argument(
        "--model",
        choices=[
            "unet",
            "u-resnet",
            "u-plusplus",
            "vit-base",
            "vit-medium",
            "vit-large",
        ],
        default="vit-medium",
        help="Model type [default: vit-medium]",
    )
    return parser.parse_args()


def get_truth_vs_predicted(
    model: Module, loader: DataLoader, device: str
) -> Tuple[Tensor, Tensor]:
    truth = Tensor().to(device)
    predicted = Tensor().to(device)

    with no_grad():
        model.eval()
        for inputs, targets in tqdm(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            outputs, targets = filter(outputs, targets)

            truth = cat((truth, targets))
            predicted = cat((predicted, outputs))

    return truth.cpu().numpy(), predicted.cpu().numpy()


if __name__ == "__main__":
    img_dir = os.getenv("IMG_DIR")
    model_dir = os.getenv("MODEL_DIR")
    patch_dir = os.getenv("PATCH_DIR")
    results_dir = os.getenv("RESULTS_DIR")
    gedi_file = os.getenv("GEDI_DIR")
    image_size = 256
    random_state = 42
    num_workers = os.cpu_count() // 2
    bins = list(range(0, 55, 5))
    device = get_device()
    config = get_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Set random seed
    seed_everyting(random_state)

    batch_size = config.batch_size

    # Get data
    _, _, test_dl = get_data(
        img_dir,
        patch_dir,
        gedi_file,
        image_size,
        batch_size,
        num_workers,
        bins,
    )

    info(f"Loading model {config.model}")
    model = load(os.path.join(model_dir, config.model), device)

    # Test model
    test_loss, test_loss_by_range = test(model, test_dl, loss, device, bins)

    info(
        f"Final test loss: {test_loss:>8f}\n"
        f"Ranges: {bins}\n"
        f"Losses by range: {test_loss_by_range}"
    )

    info("Getting truth vs predicted")
    truth, pred = get_truth_vs_predicted(model, test_dl, device)
