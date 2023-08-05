from typing import Callable

from torch import device as Device
from torch import no_grad
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_loop(
    dataloader: DataLoader,
    model: Module,
    fn: Callable,
    writer: SummaryWriter,
    device: Device,
    epoch: int,
    optimizer: Optimizer,
):
    model.train()
    for batch, (inputs, targets) in enumerate(
        tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
    ):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("loss/train", loss.item(), epoch * len(dataloader) + batch)


def validation_loop(
    dataloader: DataLoader,
    model: Module,
    fn: Callable,
    writer: SummaryWriter,
    device: Device,
    epoch: int,
):
    model.eval()
    n_batches = len(dataloader)
    loss = 0

    with no_grad():
        for inputs, targets in tqdm(dataloader, desc=f"Validation Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss += fn(outputs, targets).item()

    loss /= n_batches
    writer.add_scalar("loss/validation", loss, epoch)
