from typing import List
from torch import Tensor, cat
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    LeakyReLU,
    Module,
    Sequential,
)


class DecoderBlock(Module):
    def __init__(self, in_channels: int, skip_channels: int):
        super(DecoderBlock, self).__init__()

        self.convT = ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )

        self.decode = Sequential(
            Conv2d(
                in_channels // 2 + skip_channels,
                in_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            BatchNorm2d(in_channels // 2),
            LeakyReLU(),
            Conv2d(
                in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1
            ),
            BatchNorm2d(in_channels // 2),
            LeakyReLU(),
        )

    def combine_features(self, x: Tensor, skip: Tensor) -> Tensor:
        return cat([x, skip], dim=1)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.convT(x)
        x = self.combine_features(x, skip)
        x = self.decode(x)

        return x


class UnetDecoder(Module):
    def __init__(self, in_channels: int):
        super(UnetDecoder, self).__init__()

        self.db1 = DecoderBlock(in_channels, in_channels // 2)
        self.db2 = DecoderBlock(in_channels // 2, in_channels // 4)
        self.db3 = DecoderBlock(in_channels // 4, in_channels // 8)
        self.db4 = DecoderBlock(in_channels // 8, in_channels // 16)

    def forward(self, x: Tensor, skips: List[Tensor]) -> Tensor:
        x = self.db1(x, skips[-1])
        x = self.db2(x, skips[-2])
        x = self.db3(x, skips[-3])
        x = self.db4(x, skips[-4])

        return x
