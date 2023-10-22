from logging import info
from math import sqrt
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from numpy import ndarray
from torch import (
    Tensor,
    from_numpy,
    isnan,
    no_grad,
    stack,
    where,
    zeros,
)
from torch import device as Device
from torch import load as tload
from torch import save as tsave
from torch.nn import L1Loss, Module, MSELoss, SmoothL1Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .loss import filter
from .loss import loss_by_range as range_loss
from .plots import brighten


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
    ranges: Optional[List[int]] = list(range(0, 55, 5)),
) -> float:
    # Set model to evaluation mode
    model.eval()

    # Create pairwise tuples or ranges from bins
    range_bins = list(zip(ranges[:-1], ranges[1:]))
    loss_by_range: Tensor = zeros(len(range_bins)).to(device)

    # Huber loss
    loss: float = 0

    # MAE loss
    mae_loss: float = 0
    mae = L1Loss()

    # RMSE loss
    rmse_loss: float = 0
    mse = MSELoss()

    with no_grad():
        for _, (inputs, targets) in enumerate(
            tqdm(loader, f"Validation epoch {epoch + 1}")
        ):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            filtered_outputs, filtered_targets = filter(outputs, targets)

            loss += criterion(filtered_outputs, filtered_targets).item()
            mae_loss += mae(filtered_outputs, filtered_targets).item()
            rmse_loss += sqrt(mse(filtered_outputs, filtered_targets).item())

            batch_loss_by_range = range_loss(outputs, targets, range_bins)
            loss_by_range += where(isnan(batch_loss_by_range), 0, batch_loss_by_range)

    # Average losses
    loss /= len(loader)
    mae_loss /= len(loader)
    rmse_loss /= len(loader)
    loss_by_range /= len(loader)

    # Add loss to writer
    writer.add_scalar("Loss/val/total", loss, epoch)
    writer.add_scalar("Loss/val/MAE", mae_loss, epoch)
    writer.add_scalar("Loss/val/RMSE", rmse_loss, epoch)

    # Add loss_by_range to writer
    loss_dict = {
        f"{lower}-{upper}": loss
        for (lower, upper), loss in zip(range_bins, loss_by_range.cpu().numpy())
    }
    writer.add_scalars("Loss/val/range", loss_dict, epoch)

    # Add images to writer
    # Since the images never change, only add them once
    if epoch == 0:
        writer.add_images(
            "Plots/images",
            brighten(inputs[:, :3, :, :].cpu().numpy()),
            epoch,
            dataformats="NCHW",
        )

    # Add predictions to writer
    preds = stack([apply_colormap(output) for output in outputs])

    writer.add_images("Plots/predictions", preds, epoch, dataformats="NHWC")

    info(
        f"Validation epoch {epoch + 1}\n"
        f"Total loss: {loss:>8f}\n"
        f"MAE loss: {mae_loss:>8f}\n"
        f"RMSE loss: {rmse_loss:>8f}\n"
        f"Losses by range: {loss_by_range.cpu().numpy()}"
    )

    return loss


def test(
    model: Module,
    loader: DataLoader,
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: Device,
    ranges: Optional[List[int]] = list(range(0, 55, 5)),
) -> Tuple[float, float, float, ndarray]:
    model.eval()

    # Create pairwise tuples or ranges from bins
    range_bins = list(zip(ranges[:-1], ranges[1:]))
    loss_by_range: Tensor = zeros(len(range_bins)).to(device)

    # Huber loss
    loss: float = 0

    # MAE loss
    mae_loss: float = 0
    mae = L1Loss()

    # RMSE loss
    rmse_loss: float = 0
    mse = MSELoss()

    with no_grad():
        for _, (inputs, targets) in enumerate(tqdm(loader, "Testing")):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            filtered_outputs, filtered_targets = filter(outputs, targets)

            loss += criterion(filtered_outputs, filtered_targets).item()
            mae_loss += mae(filtered_outputs, filtered_targets).item()
            rmse_loss += sqrt(mse(filtered_outputs, filtered_targets).item())

            batch_loss_by_range = range_loss(outputs, targets, range_bins)
            loss_by_range += where(isnan(batch_loss_by_range), 0, batch_loss_by_range)

    # Average losses
    loss /= len(loader)
    mae_loss /= len(loader)
    rmse_loss /= len(loader)
    loss_by_range /= len(loader)

    return loss, mae_loss, rmse_loss, loss_by_range.cpu().numpy()


def apply_colormap(img: Tensor) -> Tensor:
    min, max = img.min().item(), img.max().item()
    img = (img.cpu() - min) / (max - min)
    cmap = plt.cm.viridis(img.numpy())[..., :3]
    return from_numpy(cmap).float()


def load(path: str, device: Device) -> Module:
    model = tload(path, map_location=device)

    model.eval()

    return model


def save(model: Module, path: str) -> None:
    tsave(model, path)
