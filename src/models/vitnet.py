from torch.nn import BatchNorm2d, Conv2d, ConvTranspose2d, Linear, Module
from torch.nn.functional import relu
from transformers import ViTConfig, ViTModel


class VitEncoder(Module):
    def __init__(self):
        super(VitEncoder, self).__init__()

        config = ViTConfig(
            image_size=256,
            patch_size=16,
            num_channels=4,
        )

        self.encoder = ViTModel(config)

    def forward(self, x):
        x = self.encoder(x).last_hidden_state

        return x


class ConvNetDecoder(Module):
    def __init__(self):
        super(ConvNetDecoder, self).__init__()

        # Reprojection Block
        self.reprojection = Linear(768, 128)

        # Reshaping to [batch_size, 128, 16, 16]
        self.reshape = lambda x: x.view(x.size(0), 128, 16, 16)

        # Decoding Blocks
        self.deconv1 = ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv2 = ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.deconv3 = ConvTranspose2d(32, 16, 3, 2, 1, 1)
        self.deconv4 = ConvTranspose2d(16, 8, 3, 2, 1, 1)

        self.norm1 = BatchNorm2d(64)
        self.norm2 = BatchNorm2d(32)
        self.norm3 = BatchNorm2d(16)
        self.norm4 = BatchNorm2d(8)

        # Output layer to produce [batch_size, 1, 256, 256] output
        self.output_conv = Conv2d(8, 1, 1, 1, 0)

    def forward(self, x):
        # Take the first 256 tokens (ignoring the class token)
        x = x[:, :256, :]

        # Reprojection
        x = self.reprojection(x)

        # Reshape
        x = self.reshape(x)

        # Decoding
        x = relu(self.norm1(self.deconv1(x)))
        x = relu(self.norm2(self.deconv2(x)))
        x = relu(self.norm3(self.deconv3(x)))
        x = relu(self.norm4(self.deconv4(x)))

        # Output layer
        x = self.output_conv(x)

        return x


class VitNet(Module):
    def __init__(self):
        super(VitNet, self).__init__()

        self.name = "vitnet"

        self.encoder = VitEncoder()
        self.decoder = ConvNetDecoder()

    def forward(self, x):
        outputs = self.encoder(x)

        outputs = self.decoder(outputs)

        return outputs
