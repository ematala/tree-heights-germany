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

    outputs, targets, mask = outputs.squeeze(), targets.squeeze(), mask.squeeze()

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
    bins: List[Tuple[int, int]],
    placeholder: float = nan,
) -> Tensor:
    losses = zeros(len(bins))

    for idx, range in enumerate(bins):
        filtered_outputs, filtered_targets = filter(outputs, targets, range)

        losses[idx] = (
            loss(filtered_outputs, filtered_targets)
            if filtered_targets.numel() > 0
            else placeholder
        )

    return losses
