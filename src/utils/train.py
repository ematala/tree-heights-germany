import logging
import os
from argparse import ArgumentParser
from logging import info
from typing import Tuple

import requests
from torch import rand
from torch.nn import Module
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from ..models import ResidualUnet, Unet, UnetPlusPlus, VitNet
from . import (
    EarlyStopping,
    get_data,
    get_device,
    loss,
    save,
    seed_everyting,
    test,
    train,
)


def get_optimizer(model: Module, lr: float) -> Optimizer:
    """
    Get optimizer for model.

    Args:
        model (Module): The model to get the optimizer for.
        lr (float): The learning rate to use for the optimizer.

    Raises:
        ValueError: If the model type is not supported.

    Returns:
        Optimizer: The optimizer for the given model.
    """
    if isinstance(model, (Unet, ResidualUnet, UnetPlusPlus)):
        return SGD(model.parameters(), lr)
    elif isinstance(model, VitNet):
        return AdamW(model.parameters(), lr)
    else:
        raise ValueError(f"Model type {type(model).__name__} not supported")


def get_model_and_optimizer(model: str = "unet") -> Tuple[Module, Optimizer]:
    """Get model and optimizer configuration.

    Args:
        model (str, optional): The model to use. Defaults to "unet".

    Raises:
        ValueError: If the model is not supported.

    Returns:
        Tuple[Module, Optimizer]: The model and optimizer instances.
    """
    config = {
        "unet": {"class": Unet, "params": {}, "lr": 1e-2},
        "u-resnet": {"class": ResidualUnet, "params": {}, "lr": 1e-2},
        "u-plusplus": {"class": UnetPlusPlus, "params": {}, "lr": 1e-2},
        "vit-base": {
            "class": VitNet,
            "params": {
                "num_attention_heads": 8,
                "hidden_size": 128,
                "intermediate_size": 512,
            },
            "lr": 1e-4,
        },
        "vit-medium": {
            "class": VitNet,
            "params": {
                "num_attention_heads": 12,
                "hidden_size": 192,
                "intermediate_size": 768,
            },
            "lr": 1e-4,
        },
        "vit-large": {
            "class": VitNet,
            "params": {
                "num_attention_heads": 16,
                "hidden_size": 256,
                "intermediate_size": 1024,
            },
            "lr": 1e-4,
        },
    }

    if model not in config:
        raise ValueError(f"Model {model} not supported")

    model_info = config[model]
    model_instance = model_info["class"](**model_info["params"])
    optimizer_instance = get_optimizer(model_instance, model_info["lr"])

    return model_instance, optimizer_instance


def get_args():
    """Get arguments from command line

    Returns:
        Namespace: Arguments
    """
    parser = ArgumentParser(
        description="Train a selected model on predicting tree canopy heights"
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
    parser.add_argument(
        "--epochs", type=int, default=25, help="Training epochs [default: 25]"
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

    return parser.parse_args()


if __name__ == "__main__":
    img_dir = os.getenv("IMG_DIR")
    log_dir = os.getenv("LOG_DIR")
    model_dir = os.getenv("MODEL_DIR")
    patch_dir = os.getenv("PATCH_DIR")
    results_dir = os.getenv("RESULTS_DIR")
    gedi_file = os.getenv("GEDI_DIR")
    num_channels = 5
    image_size = 256
    random_state = 42
    num_workers = os.cpu_count() // 2
    bins = list(range(0, 55, 5))
    device = get_device()
    config = get_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Set random seed
    seed_everyting(random_state)

    epochs = config.epochs
    batch_size = config.batch_size

    info(f"Starting training with {config.model} configuration for {epochs} epochs")

    info(f"Using device: {device}")

    # Get data
    train_dl, val_dl, test_dl = get_data(
        img_dir,
        patch_dir,
        gedi_file,
        image_size,
        batch_size,
        num_workers,
        bins,
    )

    # Create model and optimizer
    model, optimizer = get_model_and_optimizer(config.model)

    info(f"Learnable params: {model.count_params():,}")

    # Move model to device
    model.to(device)

    # Create scheduler
    scheduler = CosineAnnealingLR(optimizer, epochs)

    # Create writer
    writer = SummaryWriter(log_dir)

    # Create early stopping
    stopper = EarlyStopping()

    # Add model graph to writer
    writer.add_graph(
        model, rand(batch_size, num_channels, image_size, image_size).to(device)
    )

    # Training loop
    for epoch in range(epochs):
        info(f"Epoch {epoch + 1}\n-------------------------------")
        train(model, train_dl, loss, device, epoch, optimizer, scheduler, writer)
        val_loss, _ = test(model, val_dl, loss, device, epoch, writer)
        stopper(val_loss)
        if stopper.stop:
            info(f"Early stopping at epoch {epoch + 1}")
            break

    # Close writer
    writer.close()

    info("Training finished.")

    # Test model
    test_loss, test_loss_by_range = test(model, test_dl, loss, device, ranges=bins)

    info(
        f"Final test loss: {test_loss:>8f}\n"
        f"Ranges: {bins}\n"
        f"Losses by range: {test_loss_by_range}"
    )

    info(f"Saving model {config.model}")

    # Save model
    save(model, os.path.join(model_dir, f"{config.model}-{model.name}-{epochs}.pt"))

    if config.notify:
        token = os.getenv("TELEGRAM_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not token or not chat_id:
            raise ValueError("Telegram token and chat id must be set")

        info("Sending notification")

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": (
                f"Finished training\n"
                f"Test loss: {test_loss:>8f}\n"
                f"Losses by range: {test_loss_by_range}"
            ),
        }

        res = requests.post(url, data=data)

        info(f"Response status: {res.status_code}")
