from torch.nn import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    LeakyReLU,
    Module,
    Sequential,
)

from src.models.vitnet.reprojection import RB


class DB(Module):
    def __init__(self, in_channels: int, index: int):
        super(DB, self).__init__()

        # reprojection
        self.reproject = RB(in_channels, 32, 32, index)

        # decoding
        self.decode = Sequential(
            BatchNorm2d(in_channels),
            LeakyReLU(),
            Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(in_channels),
            LeakyReLU(),
            Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(in_channels),
            LeakyReLU(),
        )

        # upscaling
        self.upscale = Sequential(
            ConvTranspose2d(in_channels, in_channels // 4, kernel_size=4, stride=4),
            LeakyReLU(),
            Conv2d(in_channels // 4, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, prev=None):
        x = self.reproject(x)
        x = self.decode(x)
        if prev:
            x += prev
        return self.upscale(x)


class VitDecoder(Module):
    def __init__(self, in_channels: int):
        super(VitDecoder, self).__init__()

        self.db1 = DB(in_channels, 0)
        self.db2 = DB(in_channels, 1)
        self.db3 = DB(in_channels, 2)
        self.db4 = DB(in_channels, 3)

    def forward(self, tb3, tb6, tb9, tb12):
        x1 = self.db1(tb3)
        x2 = self.db2(tb6, x1)
        x3 = self.db3(tb9, x2)
        x4 = self.db4(tb12, x3)

        return x4
