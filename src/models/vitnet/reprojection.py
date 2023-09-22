from torch.nn import Conv2d, ConvTranspose2d, Module, Sequential


def blocks(in_channels: int, F: int, D: int):
    return [
        Sequential(
            Conv2d(in_channels, F, kernel_size=1, stride=1),
            Conv2d(F, D, kernel_size=3, stride=2, padding=1),
            Conv2d(D, D, kernel_size=3, stride=1, padding=1),
        ),
        Sequential(
            Conv2d(in_channels, F, kernel_size=1, stride=1),
            Conv2d(F, D, kernel_size=3, stride=1, padding=1),
        ),
        Sequential(
            Conv2d(in_channels, F, kernel_size=1, stride=1),
            ConvTranspose2d(F, F, kernel_size=2, stride=2),
            Conv2d(F, D, kernel_size=3, stride=1, padding=1),
        ),
        Sequential(
            Conv2d(in_channels, F, kernel_size=1, stride=1),
            ConvTranspose2d(F, F, kernel_size=4, stride=4),
            Conv2d(F, D, kernel_size=3, stride=1, padding=1),
        ),
    ]


class RB(Module):
    def __init__(self, in_channels: int, F: int, D: int, index: int):
        super(RB, self).__init__()
        self.block = blocks(in_channels, F, D)[index]

    def forward(self, x):
        return self.block(x)
