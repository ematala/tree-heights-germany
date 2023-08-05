from typing import Callable

from torch import device as Device
from torch import no_grad
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def training(
    dataloader: DataLoader,
    model: Module,
    fn: Callable,
    device: Device,
    writer: SummaryWriter,
    epoch: int,
    optimizer: Optimizer,
) -> None:
    model.train()
    desc = f"Training Epoch {epoch+1}"

    for batch, (inputs, targets) in enumerate(tqdm(dataloader, desc)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("loss/train", loss.item(), epoch * len(dataloader) + batch)


def evaluation(
    dataloader: DataLoader,
    model: Module,
    fn: Callable,
    device: Device,
    writer: SummaryWriter | None = None,
    epoch: int | None = None,
) -> float:
    model.eval()
    loss = 0
    desc = f"Validation Epoch {epoch+1}" if writer is not None else "Evaluation"

    with no_grad():
        for inputs, targets in tqdm(dataloader, desc):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss += fn(outputs, targets).item()

    loss /= len(dataloader)

    if writer is not None:
        writer.add_scalar("loss/validation", loss, epoch)

    return loss
