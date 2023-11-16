import torch.nn as nn
from torch import Tensor

from .blocks import (
    make_output_conv,
    make_refinenets,
)
from .decoder import make_decoder
from .encoder import make_encoder


class Vit(nn.Module):
    def __init__(
        self,
        backbone="vitb16_256",
        features=128,
        blocks=4,
        readout_op="ignore",
        use_bn=True,
        enable_attention_hooks=False,
    ):
        super(Vit, self).__init__()

        self.encoder = make_encoder(backbone, readout_op, enable_attention_hooks)
        self.decoder = make_decoder(backbone, features)
        self.refinenets = make_refinenets(features, blocks, use_bn)
        self.output_conv = make_output_conv(features)

    def forward(self, x: Tensor) -> Tensor:
        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        path = None

        for layer, refine in zip(reversed(decoded), self.refinenets):
            path = refine(layer) if path is None else refine(path, layer)

        out = self.output_conv(path)

        return out.squeeze()
