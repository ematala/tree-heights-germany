from torch import Tensor

from ..base import BaseModel
from .blocks import (
    make_output_conv,
    make_refinenets,
)
from .decoder import make_decoder
from .encoder import make_encoder


class Vit(BaseModel):
    def __init__(
        self,
        backbone="vit_tiny_patch16_256",
        embed_dim=192,
        img_size=256,
        in_chans=5,
        blocks=4,
        readout_op="ignore",
        use_bn=True,
        enable_attention_hooks=False,
        **kwargs,
    ):
        super(Vit, self).__init__()
        self.encoder = make_encoder(
            backbone,
            embed_dim,
            readout_op,
            enable_attention_hooks,
            img_size=img_size,
            in_chans=in_chans,
            **kwargs,
        )
        self.decoder = make_decoder(embed_dim, blocks)
        self.refinenets = make_refinenets(embed_dim, blocks, use_bn)
        self.output_conv = make_output_conv(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        path = None

        for layer, refine in zip(reversed(decoded), self.refinenets):
            path = refine(layer) if path is None else refine(path, layer)

        out = self.output_conv(path)

        return out.squeeze()
