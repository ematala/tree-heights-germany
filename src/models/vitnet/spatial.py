from typing import Tuple

from torch import Tensor
from torch.nn import Module


class SpatialConcatenation(Module):
    def __init__(self, input_shape: Tuple[int, int], output_shape: Tuple[int, int]):
        super(SpatialConcatenation, self).__init__()
        self.num_tokens, self.feature_dim = input_shape
        self.height, self.width = output_shape

        assert (
            self.num_tokens == self.width * self.height
        ), "Input and output shapes are not compatible"

    def forward(self, x: Tensor) -> Tensor:
        # Permute from (batch, tokens, features) to (batch, features, tokens)
        x = x.permute(0, 2, 1)

        # Reshape to obtain the spatial dimensions
        x = x.reshape(-1, self.height, self.width, self.feature_dim)

        # Permute from (batch, height, width, features) to (batch, features, height, width)
        x = x.permute(0, 3, 1, 2)

        return x
