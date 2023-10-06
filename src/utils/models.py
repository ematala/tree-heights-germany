from typing import Callable, Optional

from torch import Tensor, no_grad
from torch import device as Device
from torch import load as tload
from torch import save as tsave
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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
            print(f"Train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            if writer:
                writer.add_scalar("Loss/train", loss, epoch * len(loader) + batch)


def test(
    model: Module,
    loader: DataLoader,
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: Device,
    epoch: Optional[int] = None,
    writer: Optional[SummaryWriter] = None,
) -> float:
    model.eval()

    loss: float = 0

    with no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            loss += criterion(outputs, targets).item()

    loss /= len(loader)

    print(f"Test loss: {loss:>8f}")
    if writer and epoch:
        writer.add_scalar("Loss/test", loss, epoch)

    return loss


def load(path: str, device: Device) -> Module:
    model = tload(path, map_location=device)

    model.eval()

    return model


def save(model: Module, path: str) -> None:
    tsave(model, path)
