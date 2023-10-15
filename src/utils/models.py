from logging import info
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
from torch import (
    Tensor,
    from_numpy,
    histogram,
    isnan,
    no_grad,
    stack,
    where,
    zeros,
)
from torch import device as Device
from torch import load as tload
from torch import save as tsave
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .loss import loss_by_range as range_loss


def train(
    model: Module,
    loader: DataLoader,
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: Device,
    epoch: int,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler] = None,
    writer: Optional[SummaryWriter] = None,
) -> None:
    model.train()

    for batch, (inputs, targets) in enumerate(tqdm(loader, desc="Training")):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if scheduler:
            scheduler.step()

        if batch % 10 == 0 and writer:
            writer.add_scalar("Loss/train", loss.item(), epoch * len(loader) + batch)


def test(
    model: Module,
    loader: DataLoader,
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: Device,
    epoch: Optional[int] = None,
    writer: Optional[SummaryWriter] = None,
    ranges: Optional[List[int]] = list(range(0, 55, 5)),
) -> Tuple[float, Tensor]:
    model.eval()

    # Create pairwise tuples or ranges from bins
    range_bins = list(zip(ranges[:-1], ranges[1:]))

    loss: float = 0
    loss_by_range: Tensor = zeros(len(range_bins))

    with no_grad():
        for _, (inputs, targets) in enumerate(tqdm(loader, desc="Testing")):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            loss += criterion(outputs, targets).item()

            batch_loss_by_range = range_loss(outputs, targets, range_bins)
            loss_by_range += where(isnan(batch_loss_by_range), 0, batch_loss_by_range)

    loss /= len(loader)
    loss_by_range /= len(loader)

    info(f"Test loss: {loss:>8f}\nLosses by range: {loss_by_range.numpy()}")

    if writer and epoch is not None:
        # Add loss to writer
        writer.add_scalar("Loss/test", loss, epoch)

        # Add prediction histogram to writer
        hist = histogram(outputs.cpu()).hist
        writer.add_histogram("Stats/predictions", hist, epoch, ranges)

        # Add images to writer
        writer.add_images(
            "Plots/images", inputs[:, :3, :, :], epoch, dataformats="NCHW"
        )

        # Add predictions to writer
        preds = stack([apply_colormap(output) for output in outputs])

        writer.add_images("Plots/predictions", preds, epoch, dataformats="NHWC")

        # Add losses by range to writer
        writer.add_scalars(
            "Loss/test",
            {
                f"range-{lower}-{upper}": loss.item()
                for (lower, upper), loss in zip(range_bins, loss_by_range)
            },
            epoch,
        )

    return loss, loss_by_range.numpy()


def apply_colormap(img: Tensor) -> Tensor:
    return from_numpy(plt.cm.viridis(img.cpu().numpy())[..., :3]).float()


def load(path: str, device: Device) -> Module:
    model = tload(path, map_location=device)

    model.eval()

    return model


def save(model: Module, path: str) -> None:
    tsave(model, path)
