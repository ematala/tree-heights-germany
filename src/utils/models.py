from logging import info
from typing import Callable, List, Optional, Tuple

from torch import Tensor, isnan, no_grad, where, zeros
from torch import device as Device
from torch import load as tload
from torch import save as tsave
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .loss import loss_by_range


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
    size = len(loader.dataset)

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

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(inputs)
            info(f"Train loss: {loss:>8f}  [{current:>5d}/{size:>5d}]")
            if writer:
                writer.add_scalar("Loss/train", loss, epoch * len(loader) + batch)


def test(
    model: Module,
    loader: DataLoader,
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: Device,
    epoch: Optional[int] = None,
    writer: Optional[SummaryWriter] = None,
    bins: Optional[List[int]] = list(range(0, 55, 5)),
) -> Tuple[float, Tensor, List[Tuple[int, int]]]:
    model.eval()

    # Create pairwise tuples or ranges from bins
    bins = list(zip(bins[:-1], bins[1:]))

    loss: float = 0
    losses_by_range: Tensor = zeros(len(bins))

    with no_grad():
        for _, (inputs, targets) in enumerate(tqdm(loader, desc="Testing")):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            loss += criterion(outputs, targets).item()

            batch_loss_by_range = loss_by_range(outputs, targets, bins)
            losses_by_range += where(isnan(batch_loss_by_range), 0, batch_loss_by_range)

    loss /= len(loader)
    losses_by_range /= len(loader)

    info(f"Test loss: {loss:>8f}\nLosses by range: {losses_by_range.numpy()}")

    if writer and epoch:
        writer.add_scalar("Loss/test", loss, epoch)

        ranges = list(zip(bins[:-1], bins[1:]))
        for idx, range in enumerate(ranges):
            lower, upper = range
            writer.add_scalar(
                f"Loss/test/range-{lower}-{upper}", losses_by_range[idx].item(), epoch
            )

    return loss, losses_by_range.numpy(), bins


def load(path: str, device: Device) -> Module:
    model = tload(path, map_location=device)

    model.eval()

    return model


def save(model: Module, path: str) -> None:
    tsave(model, path)
