from typing import List, Optional, Tuple

from torch import Tensor, zeros, nan
from torch.nn import HuberLoss


def filter(
    outputs: Tensor,
    targets: Tensor,
    range: Optional[Tuple[int, int]] = None,
    no_data: float = 0,
) -> Tuple[Tensor, Tensor]:
    mask = targets != no_data

    if range:
        lower, upper = range
        mask &= (targets >= lower) & (targets < upper)

    return outputs[mask], targets[mask]


def loss(
    outputs: Tensor,
    targets: Tensor,
    delta: float = 3,
) -> Tensor:
    outputs, targets = filter(outputs, targets)

    fn = HuberLoss("mean", delta)

    return fn(outputs, targets)


def range_loss(
    outputs: Tensor,
    targets: Tensor,
    ranges: List[int] = list(range(0, 55, 5)),
) -> Tensor:
    ranges = list(zip(ranges[:-1], ranges[1:]))

    losses = zeros(len(ranges))

    for i, range in enumerate(ranges):
        outputs, targets = filter(outputs, targets, range)

        losses[i] = loss(outputs, targets) if targets.numel() > 0 else nan

    return losses
