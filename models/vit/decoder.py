from typing import List

import torch.nn as nn


class VitDecoder(nn.Module):
    def __init__(
        self,
        in_shape: List[int],
        features: int,
        groups=1,
        expand=False,
    ) -> None:
        super().__init__()
        out_shape1 = features
        out_shape2 = features
        out_shape3 = features
        out_shape4 = features

        if expand:
            out_shape1 = features
            out_shape2 = features * 2
            out_shape3 = features * 4
            out_shape4 = features * 8

        self.layer1_rn = nn.Conv2d(
            in_shape[0],
            out_shape1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )

        self.layer2_rn = nn.Conv2d(
            in_shape[1],
            out_shape2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )

        self.layer3_rn = nn.Conv2d(
            in_shape[2],
            out_shape3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )

        self.layer4_rn = nn.Conv2d(
            in_shape[3],
            out_shape4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )


def make_decoder(
    backbone,
    features,
    groups=1,
    expand=False,
):
    return {
        "vitb16_384": VitDecoder(
            [16, 32, 64, 128], features, groups, expand
        ),  # ViT-B/16 - 84.6% Top1 (backbone),
        "vitl16_384": VitDecoder(
            [256, 512, 1024, 1024], features, groups, expand
        ),  # ViT-L/16 - 85.0% Top1 (backbone)
        "vitb_rn50_384": VitDecoder(
            [256, 512, 768, 768], features, groups, expand
        ),  # ViT-H/16 - 85.0% Top1 (backbone)
    }[backbone]
