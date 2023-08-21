from typing import List, Tuple

from torch import Tensor, abs


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
) -> Tensor:
    outputs, targets = filter(outputs, targets)

    return abs(targets - outputs).mean()


def range_loss(outputs: Tensor, targets: Tensor, lower: int, upper: int) -> float:
    outputs, targets = filter(outputs, targets)

    idx = (targets >= lower) & (targets < upper)

    return abs(targets[idx] - outputs[idx]).mean().item()


def range_losses(
    outputs: Tensor, targets: Tensor, ranges: List[Tuple[int, int]]
) -> List[float]:
    return [range_loss(outputs, targets, lower, upper) for lower, upper in ranges]
