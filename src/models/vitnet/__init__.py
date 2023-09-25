from torch import Tensor
from torch.nn import Module
from transformers import ViTConfig

from .decoder import VitDecoder
from .encoder import VitEncoder
from .output import OutputHead


class VitNet(Module):
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        num_channels: int = 4,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 8,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
    ):
        super(VitNet, self).__init__()

        self.name = f"vitnet-{hidden_size}"

        self.config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            output_hidden_states=True,
        )

        self.encoder = VitEncoder(self.config)

        self.decoder = VitDecoder(
            input_shape=((image_size // patch_size) ** 2, hidden_size),
            output_shape=(image_size // patch_size, image_size // patch_size),
        )

        self.head = OutputHead(hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        tb3, tb6, tb9, tb12 = self.encoder(x)

        output = self.decoder(tb3, tb6, tb9, tb12)

        return self.head(output)
