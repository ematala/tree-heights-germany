import logging
from math import sqrt
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from numpy import ndarray
from torch import (
    Tensor,
    cat,
    from_numpy,
    no_grad,
    stack,
    zeros,
)
from torch import device as Device
from torch.nn import L1Loss, Module, MSELoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .loss import filter, loss_by_range
from .plots import brighten_image
from .transforms import denormalize, scale


def train(
    model: Module,
    loader: DataLoader,
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: Device,
    epoch: int,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler] = None,
    writer: Optional[SummaryWriter] = None,
    teacher: Optional[Module] = None,
    alpha: float = 0.5,
) -> None:
    model.train()

    for batch, (inputs, targets) in enumerate(
        tqdm(loader, f"Training epoch {epoch + 1}")
    ):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        loss = criterion(*filter(outputs, targets))

        if teacher:
            with no_grad():
                teacher_outputs = teacher(inputs)

            loss = (1 - alpha) * loss + alpha * criterion(outputs, teacher_outputs)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if scheduler:
            scheduler.step()

        if batch % 10 == 0 and writer:
            step = epoch * len(loader) + batch
            writer.add_scalar("Loss/train/total", loss.item(), step)


def validate(
    model: Module,
    loader: DataLoader,
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: Device,
    epoch: int,
    writer: SummaryWriter,
) -> float:
    # Set model to evaluation mode
    model.eval()

    # define loss functions
    mae = L1Loss()
    mse = MSELoss()

    metrics = {
        "total": 0,
        "mae": 0,
        "rmse": 0,
    }

    with no_grad():
        for _, (inputs, targets) in enumerate(
            tqdm(loader, f"Validation epoch {epoch + 1}")
        ):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            filtered_outputs, filtered_targets = filter(outputs, targets)

            metrics["total"] += criterion(filtered_outputs, filtered_targets).item()
            metrics["mae"] += mae(filtered_outputs, filtered_targets).item()
            metrics["rmse"] += sqrt(mse(filtered_outputs, filtered_targets).item())

    # Average losses
    for key in metrics.keys():
        metrics[key] /= len(loader)

    # Add loss to writer
    writer.add_scalar("Loss/val/total", metrics.get("total"), epoch)
    writer.add_scalar("Loss/val/MAE", metrics.get("mae"), epoch)
    writer.add_scalar("Loss/val/RMSE", metrics.get("rmse"), epoch)

    # Add images to writer
    # Since the images never change, only add them once
    if epoch == 0:
        images = torch.stack([scale(denormalize(batch)) for batch in inputs])
        images = brighten_image(images[:, :3, :, :].cpu().numpy())
        writer.add_images("Plots/images", images, epoch, dataformats="NCHW")

    # Add predictions to writer
    preds = stack([apply_colormap(output) for output in outputs])

    writer.add_images("Plots/predictions", preds, epoch, dataformats="NHWC")

    logging.info(
        f"Validation epoch {epoch + 1}\n"
        f"Total loss: {metrics.get('total'):>8f}\n"
        f"MAE loss: {metrics.get('mae'):>8f}\n"
        f"RMSE loss: {metrics.get('rmse'):>8f}\n"
    )

    return metrics.get("total")


def test(
    model: Module,
    loader: DataLoader,
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: Device,
    ranges: Optional[List[int]] = list(range(0, 55, 5)),
) -> Tuple[float, float, float, ndarray, ndarray, ndarray]:
    model.eval()

    # Create pairwise tuples or ranges from bins
    range_bins = list(zip(ranges[:-1], ranges[1:]))

    metrics = {
        "total": 0,
        "mae": 0,
        "rmse": 0,
        "loss_by_range": zeros(len(range_bins)).to(device),
    }

    predictions = {
        "targets": [],
        "predicted": [],
    }

    # define loss functions
    mae = L1Loss()
    mse = MSELoss()

    with no_grad():
        for _, (inputs, targets) in enumerate(tqdm(loader, "Testing")):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            filtered_outputs, filtered_targets = filter(outputs, targets)

            metrics["total"] += criterion(filtered_outputs, filtered_targets).item()
            metrics["mae"] += mae(filtered_outputs, filtered_targets).item()
            metrics["rmse"] += sqrt(mse(filtered_outputs, filtered_targets).item())

            metrics["loss_by_range"] += loss_by_range(
                outputs, targets, range_bins, criterion
            )

            predictions["targets"].append(filtered_targets.cpu())
            predictions["predicted"].append(filtered_outputs.cpu())

    # Average losses
    for key in metrics.keys():
        metrics[key] /= len(loader)

    metrics["loss_by_range"] = metrics.get("loss_by_range").cpu().numpy()

    # Convert to numpy arrays
    for key in predictions.keys():
        predictions[key] = cat(predictions.get(key)).cpu().numpy()

    return dict(**metrics, **predictions)


def apply_colormap(img: Tensor) -> Tensor:
    min, max = img.min().item(), img.max().item()
    img = (img.cpu() - min) / (max - min)
    cmap = plt.cm.viridis(img.numpy())[..., :3]
    return from_numpy(cmap).float()
