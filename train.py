import argparse
import logging
import os

from dotenv import load_dotenv
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from models import get_all_models, make_model
from utils.loss import loss
from utils.misc import (
    get_device,
    get_num_processes_to_spawn,
    seed_everyting,
    send_telegram_message,
)
from utils.models import test, train, validate
from utils.pipeline import get_data
from utils.stopping import EarlyStopping


def main():
    args = get_training_args()
    image_size = 256
    random_state = 42
    num_workers = get_num_processes_to_spawn()
    bins = list(range(0, 55, 5))
    device = get_device()
    config = vars(args)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Set random seed
    seed_everyting(random_state)

    epochs = config.get("epochs")
    batch_size = config.get("batch_size")
    alpha = config.get("alpha")
    patience = config.get("patience")
    img_dir = config.get("img_dir")
    log_dir = config.get("log_dir")
    weights_dir = config.get("weights_dir")
    patch_dir = config.get("patch_dir")
    gedi_dir = config.get("gedi_dir")
    model_name = config.get("model")
    teacher_name = config.get("teacher")
    use_mp = config.get("use_mp") and device.type == "cuda"

    logging.info(
        f"Starting training...\n"
        f"Configuration: {model_name}\n"
        f"Epochs: {epochs}\n"
        f"Device: {device}\n"
        f"Batch size: {batch_size}\n"
        f"Mixed precision training: {use_mp}\n"
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

    # Create model and move to device
    model = make_model(model_name).to(device)

    # constant lr for all models
    lr = 1e-4

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr)

    # Create scaler for mixed precision training
    scaler = GradScaler() if use_mp else None

    # Create teacher model
    teacher = None

    if teacher_name:
        logging.info(f"Loading teacher model {teacher_name}")
        teacher = (
            make_model(teacher_name)
            .load(os.path.join(weights_dir, f"{teacher_name}.pt"))
            .to(device)
        )

    # Create scheduler
    scheduler = CosineAnnealingLR(optimizer, epochs)

    # Create writer
    writer = SummaryWriter(f"{log_dir}/{model_name}")

    # Create early stopping
    stopper = EarlyStopping(model, weights_dir, model_name, patience)

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
            scaler,
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
    metrics = test(model, test_dl, loss, device, bins)

    report = (
        f"Finished training with {model_name} configuration for {epochs} epochs\n"
        f"Early stopping triggered at epoch {trained_epochs}\n"
        f"Final test loss: {metrics.get('total'):>8f}\n"
        f"Final MAE loss: {metrics.get('mae'):>8f}\n"
        f"Final RMSE loss: {metrics.get('rmse'):>8f}\n"
        f"Ranges: {bins}\n"
        f"Losses by range: {metrics.get('loss_by_range')}"
    )

    logging.info(report)

    if config.get("notify"):
        send_telegram_message(report)


def get_training_args():
    """Get arguments from command line

    Returns:
        Namespace: Arguments
    """
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Train a selected model on predicting tree canopy heights"
    )

    parser.add_argument(
        "--img_dir",
        type=str,
        default=os.getenv("IMG_DIR", "data/images"),
        help="Path to images directory [default: data/images]",
    )
    parser.add_argument(
        "--patch_dir",
        type=str,
        default=os.getenv("PATCH_DIR", "data/patches"),
        help="Path to patches directory [default: data/patches]",
    )
    parser.add_argument(
        "--gedi_dir",
        type=str,
        default=os.getenv("GEDI_DIR", "data/gedi"),
        help="Path to GEDI directory [default: data/gedi]",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=os.getenv("WEIGHTS_DIR", "weights"),
        help="Path to weights directory [default: weights]",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=os.getenv("LOG_DIR", "logs"),
        help="Path to logs directory [default: logs]",
    )

    parser.add_argument(
        "--model",
        choices=get_all_models(),
        default="vit-tiny",
        help="Model type [default: vit-tiny]",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Training epochs [default: 50]"
    )

    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size [default: 64]"
    )

    parser.add_argument(
        "--notify",
        action="store_true",
        default=True,
        help="Send telegram notification when training is finished",
    )

    parser.add_argument(
        "--teacher",
        choices=get_all_models(),
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

    parser.add_argument(
        "--use_mp",
        action="store_true",
        default=True,
        help="Use mixed precision training",
    )

    parser.add_argument(
        "--no_use_mp",
        action="store_false",
        dest="use_mp",
        help="Do not use mixed precision training",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
