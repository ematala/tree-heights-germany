from typing import Tuple

from torch import Tensor
from torch.nn import Conv2d, ConvTranspose2d, Module, Sequential

from .spatial import SpatialConcatenation


def get_reprojection_block(
    index: int,
    input_shape: Tuple[int, int],
    num_channels: int,
) -> Sequential:
    _, feature_dim = input_shape

    return [
        Sequential(
            Conv2d(feature_dim, num_channels, kernel_size=1, stride=1),
            Conv2d(num_channels, feature_dim, kernel_size=3, stride=2, padding=1),
            Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1),
        ),
        Sequential(
            Conv2d(feature_dim, num_channels, kernel_size=1, stride=1),
            Conv2d(num_channels, feature_dim, kernel_size=3, stride=1, padding=1),
        ),
        Sequential(
            Conv2d(feature_dim, num_channels, kernel_size=1, stride=1),
            ConvTranspose2d(num_channels, num_channels, kernel_size=2, stride=2),
            Conv2d(num_channels, feature_dim, kernel_size=3, stride=1, padding=1),
        ),
        Sequential(
            Conv2d(feature_dim, num_channels, kernel_size=1, stride=1),
            ConvTranspose2d(num_channels, num_channels, kernel_size=4, stride=4),
            Conv2d(num_channels, feature_dim, kernel_size=3, stride=1, padding=1),
        ),
    ][index]


class ReprojectionBlock(Module):
    def __init__(
        self,
        index: int,
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
        num_channels: int = 64,
    ):
        super(ReprojectionBlock, self).__init__()

        self.spatial = SpatialConcatenation(input_shape, output_shape)

        self.block = get_reprojection_block(index, input_shape, num_channels)

    def forward(self, x: Tensor) -> Tensor:
        # spatial concatenation
        x = self.spatial(x)
        # reprojection
        x = self.block(x)

        return x
