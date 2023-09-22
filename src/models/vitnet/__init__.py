from torch.nn import Module

from src.models.vitnet.decoder import VitDecoder
from src.models.vitnet.encoder import VitEncoder


class VitNet(Module):
    def __init__(self):
        super(VitNet, self).__init__()

        self.name = "vitnet"

        self.encoder = VitEncoder()
        self.decoder = VitDecoder(512)

    def forward(self, x):
        tb3, tb6, tb9, tb12 = self.encoder(x)

        return self.decoder(tb3, tb6, tb9, tb12)
