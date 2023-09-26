from typing import List, Tuple

from torch import Tensor
from torch.nn import BatchNorm2d, Conv2d, LeakyReLU, Module, Sequential


class EncoderBlock(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(EncoderBlock, self).__init__()

        self.encode = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_channels),
            LeakyReLU(),
            Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_channels),
            LeakyReLU(),
        )

        self.conv = Conv2d(out_channels, out_channels * 2, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        skip = self.encode(x)
        x = self.conv(skip)

        return x, skip


class UnetEncoder(Module):
    def __init__(
        self,
        in_channels: int,
    ):
        super(UnetEncoder, self).__init__()

        self.eb1 = EncoderBlock(in_channels, in_channels)
        self.eb2 = EncoderBlock(in_channels * 2, in_channels * 2)
        self.eb3 = EncoderBlock(in_channels * 4, in_channels * 4)
        self.eb4 = EncoderBlock(in_channels * 8, in_channels * 8)

        self.conv = Sequential(
            Conv2d(
                in_channels * 16, in_channels * 16, kernel_size=3, stride=1, padding=1
            ),
            Conv2d(
                in_channels * 16, in_channels * 16, kernel_size=3, stride=1, padding=1
            ),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        x, skip1 = self.eb1(x)
        x, skip2 = self.eb2(x)
        x, skip3 = self.eb3(x)
        x, skip4 = self.eb4(x)

        x = self.conv(x)

        return x, [skip1, skip2, skip3, skip4]
