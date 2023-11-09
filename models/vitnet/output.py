from torch import Tensor
from torch.nn import Conv2d, Module, Softplus


class OutputHead(Module):
    def __init__(self, in_channels: int):
        super(OutputHead, self).__init__()
        self.conv = Conv2d(in_channels, 1, kernel_size=1, stride=1)
        self.softplus = Softplus()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.softplus(x)

        return x
