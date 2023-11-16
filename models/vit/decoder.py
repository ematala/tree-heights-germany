from typing import List
from torch import Tensor

import torch.nn as nn


def make_decoder(
    backbone,
    features=128,
    groups=1,
    expand=False,
):
    return {
        "vitb16_256": VitDecoder([64, 96, 128, 128], features, groups, expand),
    }[backbone]


class VitDecoder(nn.Module):
    def __init__(
        self,
        in_shape: List[int],
        features: int,
        groups=1,
        expand=False,
        blocks=4,
    ) -> None:
        super(VitDecoder, self).__init__()

        out_shapes = (
            [features * (2**i) for i in range(blocks)]
            if expand
            else [features] * blocks
        )

        self.layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_shape[i],
                    out_channels=out_shapes[i],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    groups=groups,
                )
                for i in range(blocks)
            ]
        )

    def forward(self, skips: List[Tensor]) -> List[Tensor]:
        return [self.layers[i](skips[i]) for i in range(len(skips))]
