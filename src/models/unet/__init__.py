from torch import Tensor
from torch.nn import Module

from .attention import SelfAwareAttention
from .decoder import UnetDecoder
from .encoder import UnetEncoder
from .output import OutputHead


class Unet(Module):
    def __init__(self, in_channels: int = 4):
        super(Unet, self).__init__()

        self.name = f"unet-{in_channels}"

        self.encoder = UnetEncoder(in_channels)
        self.attention = SelfAwareAttention()
        self.decoder = UnetDecoder(in_channels * (2**4))
        self.head = OutputHead(in_channels)

    def count_params(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def forward(self, x: Tensor) -> Tensor:
        x, skips = self.encoder(x)
        x = self.attention(x)
        x = self.decoder(x, skips)
        x = self.head(x)
        x = x.squeeze()

        return x
