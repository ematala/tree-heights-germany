from typing import List

from numpy import dot
from numpy import histogram as nphist
from numpy import sum as npsum
from pandas import DataFrame, Series


def compute_sampling_weights(
    patches: DataFrame, labels: Series, bins: List[int] = list(range(0, 55, 5))
) -> Series:
    heights, _ = nphist(labels, bins)

    inv_probs = 1 / heights

    probs = inv_probs / npsum(inv_probs)

    weights = patches.bins.apply(lambda x: dot(x, probs))

    weights /= weights.sum()

    return weights
