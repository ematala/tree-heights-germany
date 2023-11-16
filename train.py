import logging
import os
from argparse import ArgumentParser

from dotenv import load_dotenv
from torch import rand
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from models import make_model
from utils import (
    EarlyStopping,
    get_data,
    get_device,
    load_model,
    loss,
    seed_everyting,
    send_telegram_message,
    test,
    train,
    validate,
)


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


def main():
    load_dotenv()
    img_dir = os.getenv("IMG_DIR")
    log_dir = os.getenv("LOG_DIR")
    model_dir = os.getenv("MODEL_DIR")
    patch_dir = os.getenv("PATCH_DIR")
    gedi_dir = os.getenv("GEDI_DIR")
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
    alpha = config.alpha
    patience = config.patience

    logging.info(
        f"Starting training with {config.model} configuration for {epochs} epochs on device {device}"
    )

    # Get data
    train_dl, val_dl, test_dl = get_data(
        img_dir,
        patch_dir,
        gedi_dir,
        image_size,
        batch_size,
        num_workers,
        bins,
    )

    # Create model and optimizer
    model = make_model(config.model)

    # constant lr for all models
    lr = 1e-4

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr)

    # Move model to device
    model.to(device)

    # Create teacher model
    teacher = None

    if config.teacher:
        logging.info(f"Loading teacher model {config.teacher}")
        teacher = load_model(os.path.join(model_dir, config.teacher), device)

    # Create scheduler
    scheduler = CosineAnnealingLR(optimizer, epochs)

    # Create writer
    writer = SummaryWriter(f"{log_dir}/{config.model}")

    # Create early stopping
    stopper = EarlyStopping(model, model_dir, config.model, patience)

    # Add model graph to writer
    writer.add_graph(
        model, rand(batch_size, num_channels, image_size, image_size).to(device)
    )

    # Initialize trained epochs
    trained_epochs = 0

    # Training loop
    for epoch in range(epochs):
        train(
            model,
            train_dl,
            loss,
            device,
            epoch,
            optimizer,
            scheduler,
            writer,
            teacher,
            alpha,
        )
        val_loss = validate(
            model,
            val_dl,
            loss,
            device,
            epoch,
            writer,
            bins,
        )
        trained_epochs += 1
        stopper(val_loss)
        if stopper.stop:
            logging.info(f"Early stopping at epoch {trained_epochs}")
            break

    # Close writer
    writer.close()

    logging.info("Training finished.")

    # Test model
    test_loss, test_mae, test_rmse, test_loss_by_range, _, _ = test(
        model, test_dl, loss, device, bins
    )

    report = (
        f"Finished training with {config.model} configuration for {epochs} epochs\n"
        f"Early stopping triggered at epoch {trained_epochs}\n"
        f"Final test loss: {test_loss:>8f}\n"
        f"Final MAE loss: {test_mae:>8f}\n"
        f"Final RMSE loss: {test_rmse:>8f}\n"
        f"Ranges: {bins}\n"
        f"Losses by range: {test_loss_by_range}"
    )

    logging.info(report)

    if config.notify:
        send_telegram_message(report)


if __name__ == "__main__":
    main()
