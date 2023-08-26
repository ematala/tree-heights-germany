import torch
from torch import nn
from torch.nn.functional import pad


# Define the Convolution Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# Define the Up-Convolution Block
class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(2 * out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# Define the U-Net Model


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.name = "u-custom"

        self.down_conv1 = ConvBlock(4, 64)
        self.down_conv2 = ConvBlock(64, 128)
        self.down_conv3 = ConvBlock(128, 256)
        self.down_conv4 = ConvBlock(256, 512)
        self.down_conv5 = ConvBlock(512, 1024)

        self.up_conv1 = UpConvBlock(1024, 512)
        self.up_conv2 = UpConvBlock(512, 256)
        self.up_conv3 = UpConvBlock(256, 128)
        self.up_conv4 = UpConvBlock(128, 64)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down_conv1(x)
        x2 = self.down_conv2(self.max_pool(x1))
        x3 = self.down_conv3(self.max_pool(x2))
        x4 = self.down_conv4(self.max_pool(x3))
        x5 = self.down_conv5(self.max_pool(x4))

        # Decoder
        x = self.up_conv1(x5, x4)
        x = self.up_conv2(x, x3)
        x = self.up_conv3(x, x2)
        x = self.up_conv4(x, x1)

        return self.final_conv(x)
