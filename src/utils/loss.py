from typing import Tuple

from torch import Tensor, abs
from torch.nn import HuberLoss


def filter(
    outputs: Tensor, targets: Tensor, no_data: float = 0
) -> Tuple[Tensor, Tensor]:
    outputs, targets = outputs.flatten(), targets.flatten()

    outputs = outputs[targets != no_data]
    targets = targets[targets != no_data]

    return outputs, targets


def loss(
    outputs: Tensor,
    targets: Tensor,
    delta: float = 3,
) -> Tensor:
    outputs, targets = filter(outputs, targets)

    fn = HuberLoss("mean", delta)

    return fn(outputs, targets)


def range_loss(outputs: Tensor, targets: Tensor, lower: int, upper: int) -> Tensor:
    outputs, targets = filter(outputs, targets)

    idx = (targets >= lower) & (targets < upper)

    return abs(targets[idx] - outputs[idx]).mean()
