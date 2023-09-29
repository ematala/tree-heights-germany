from typing import Callable

from torch import Tensor, no_grad
from torch import device as Device
from torch import load as tload
from torch import save as tsave
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    loader: DataLoader,
    model: Module,
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: Device,
    optimizer: Optimizer,
    scheduler: LRScheduler,
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

        scheduler.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(inputs)
            print(f"Train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(
    loader: DataLoader,
    model: Module,
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: Device,
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

    return loss


def load(model: Module, path: str, device: Device) -> Module:
    model.load_state_dict(tload(path, map_location=device))

    model.eval()

    return model


def save(model: Module, path: str) -> None:
    tsave(model.state_dict(), path)
