from typing import List

import torch.nn as nn
from torch import Tensor

from .backbone import make_feature_dims


def make_decoder(
    embed_dim: int,
    blocks: int = 4,
    drop_rate=0.1,
    groups=1,
    expand=False,
):
    return VitDecoder(
        in_shape=make_feature_dims(embed_dim, blocks),
        embed_dim=embed_dim,
        groups=groups,
        expand=expand,
        blocks=blocks,
        drop_rate=drop_rate,
    )


class VitDecoder(nn.Module):
    def __init__(
        self,
        in_shape: List[int],
        embed_dim: int,
        groups=1,
        expand=False,
        blocks=4,
        drop_rate=0.1,
    ) -> None:
        super(VitDecoder, self).__init__()

        out_shapes = (
            [embed_dim * (2**i) for i in range(blocks)]
            if expand
            else [embed_dim] * blocks
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

        self.dropouts = nn.ModuleList([nn.Dropout2d(drop_rate) for _ in range(blocks)])

    def forward(self, skips: List[Tensor]) -> List[Tensor]:
        return [self.dropouts[i](self.layers[i](skip)) for i, skip in enumerate(skips)]
