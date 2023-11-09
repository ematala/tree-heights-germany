from typing import List, Tuple

from torch import Tensor
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    LeakyReLU,
    Module,
    Sequential,
)

from torch.nn.functional import interpolate
from .reprojection import ReprojectionBlock


class DecoderBlock(Module):
    def __init__(
        self,
        index: int,
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
    ):
        super(DecoderBlock, self).__init__()

        _, feature_dim = input_shape

        # reprojection
        self.reproject = ReprojectionBlock(index, input_shape, output_shape)

        # decoding
        self.decode = Sequential(
            BatchNorm2d(feature_dim),
            LeakyReLU(),
            Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(feature_dim),
            LeakyReLU(),
            Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(feature_dim),
            LeakyReLU(),
        )

        # upscaling and final convolution
        self.upscale = Sequential(
            ConvTranspose2d(feature_dim, feature_dim // 4, kernel_size=4, stride=4),
            LeakyReLU(),
            Conv2d(feature_dim // 4, feature_dim, kernel_size=3, stride=1, padding=1),
        )

    def combine_features(self, x: Tensor, prev: Tensor) -> Tensor:
        _, _, height, width = x.shape
        return x + interpolate(
            prev,
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        )

    def forward(self, x: Tensor, prev: Tensor = None) -> Tensor:
        x = self.reproject(x)

        x = self.decode(x)

        if prev is not None:
            x = self.combine_features(x, prev)

        x = self.upscale(x)

        return x


class VitDecoder(Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
    ):
        super(VitDecoder, self).__init__()

        self.db1 = DecoderBlock(0, input_shape, output_shape)
        self.db2 = DecoderBlock(1, input_shape, output_shape)
        self.db3 = DecoderBlock(2, input_shape, output_shape)
        self.db4 = DecoderBlock(3, input_shape, output_shape)

    def forward(self, skips: List[Tensor]) -> List[Tensor]:
        x1 = self.db1(skips[0])
        x2 = self.db2(skips[1], x1)
        x3 = self.db3(skips[2], x2)
        x4 = self.db4(skips[3], x3)

        return [x1, x2, x3, x4]
