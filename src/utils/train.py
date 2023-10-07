import logging
import os
from argparse import ArgumentParser
from logging import info
from typing import Tuple

import requests
from torch.nn import Module
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from ..models import ResidualUnet, Unet, UnetPlusPlus, VitNet
from . import (
    get_data,
    get_device,
    loss,
    save,
    seed_everyting,
    test,
    train,
)


def get_optimizer(model: Module, lr: float) -> Optimizer:
    if isinstance(model, (Unet, ResidualUnet, UnetPlusPlus)):
        return SGD(model.parameters(), lr)
    elif isinstance(model, VitNet):
        return AdamW(model.parameters(), lr)
    else:
        raise ValueError(f"Model type {type(model).__name__} not supported")


def get_config(model: str = "unet") -> Tuple[Module, Optimizer]:
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
    image_size = 256
    random_state = 42
    num_workers = os.cpu_count() // 2
    bins = list(range(0, 55, 5))
    device = get_device()
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    seed_everyting(random_state)

    epochs = args.epochs
    batch_size = args.batch_size

    info(f"Starting training with {args.model} configuration for {epochs} epochs")

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
    model, optimizer = get_config(args.model)

    info(f"Learnable params: {model.count_params():,}")

    # Move model to device
    model.to(device)

    # Create scheduler
    scheduler = CosineAnnealingLR(optimizer, epochs)

    writer = SummaryWriter(log_dir)

    # Training loop
    for epoch in range(epochs):
        info(f"Epoch {epoch + 1}\n-------------------------------")
        train(model, train_dl, loss, device, epoch, optimizer, scheduler, writer)
        test(model, val_dl, loss, device, epoch, writer)

    # Close writer
    writer.close()

    info("Training finished.")

    # Test model
    score = test(model, test_dl, loss, device)

    info(f"Saving model {args.model}")

    # Save model
    save(model, os.path.join(model_dir, f"{args.model}-{model.name}-{epochs}.pt"))

    if args.notify:
        token = os.getenv("TELEGRAM_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not token or not chat_id:
            raise ValueError("Telegram token and chat id must be set")

        info("Sending notification")

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {"chat_id": chat_id, "text": f"Finished training\n\nTest score: {score}"}

        res = requests.post(url, data=data)

        info(f"Response status: {res.status_code}")
