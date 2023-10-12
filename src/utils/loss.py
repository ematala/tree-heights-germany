from typing import List, Optional, Tuple

from torch import Tensor, nan, zeros
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


def loss_by_range(
    outputs: Tensor,
    targets: Tensor,
    bins: List[int],
) -> Tuple[Tensor, List[Tuple[int, int]]]:
    bins = list(zip(bins[:-1], bins[1:]))

    losses = zeros(len(bins))

    for i, range in enumerate(bins):
        outputs, targets = filter(outputs, targets, range)

        losses[i] = loss(outputs, targets) if targets.numel() > 0 else nan

    return losses, bins
