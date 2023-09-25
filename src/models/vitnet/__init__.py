from torch import Tensor
from torch.nn import Module

from .decoder import VitDecoder
from .encoder import VitEncoder
from .output import OutputHead


class VitNet(Module):
    def __init__(self):
        super(VitNet, self).__init__()

        self.name = "vitnet"

        self.encoder = VitEncoder()
        self.decoder = VitDecoder((256, 512), (16, 16))
        self.head = OutputHead(512)

    def forward(self, x: Tensor) -> Tensor:
        tb3, tb6, tb9, tb12 = self.encoder(x)
        output = self.decoder(tb3, tb6, tb9, tb12)

        return self.head(output)
