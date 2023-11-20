from math import sqrt
from typing import Callable, List, Optional, Tuple

from torch import Tensor, zeros
from torch.nn import L1Loss, MSELoss, SmoothL1Loss


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
    fn = SmoothL1Loss(beta=delta)

    return fn(outputs, targets)


def loss_by_range(
    outputs: Tensor,
    targets: Tensor,
    bins: List[Tuple[int, int]],
    criterion: Callable[[Tensor, Tensor], Tensor],
    placeholder: float = 0,
) -> Tensor:
    metrics = {
        "total": zeros(len(bins)).to(outputs.device),
        "mae": zeros(len(bins)).to(outputs.device),
        "rmse": zeros(len(bins)).to(outputs.device),
    }

    mae = L1Loss()
    mse = MSELoss()

    for idx, range in enumerate(bins):
        filtered_outputs, filtered_targets = filter(outputs, targets, range)

        if filtered_targets.numel() > 0:
            metrics["total"][idx] = criterion(filtered_outputs, filtered_targets)
            metrics["mae"][idx] = mae(filtered_outputs, filtered_targets)
            metrics["rmse"][idx] = sqrt(mse(filtered_outputs, filtered_targets))
        else:
            for key in metrics.keys():
                metrics[key][idx] = placeholder

    return metrics.get("total")
