from typing import Any

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
        num_channels: int = 5,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 8,
        hidden_size: int = 128,
        intermediate_size: int = 512,
        hidden_dropout_prob: float = 0.1,
        **kwargs: Any,
    ):
        super(VitNet, self).__init__()

        if not image_size % patch_size == 0:
            raise ValueError("image_size must be divisible by patch_size")
        if not hidden_size % num_attention_heads == 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if not intermediate_size >= hidden_size:
            raise ValueError(
                "intermediate_size should be greater than or equal to hidden_size"
            )

        self.name = f"vit-{patch_size}"

        self.config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            output_hidden_states=True,
            output_attentions=True,
            **kwargs,
        )

        self.encoder = VitEncoder(self.config)

        self.decoder = VitDecoder(
            input_shape=((image_size // patch_size) ** 2, hidden_size),
            output_shape=(image_size // patch_size, image_size // patch_size),
        )

        self.head = OutputHead(hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        skips = self.encoder(x)

        _, _, _, x4 = self.decoder(skips)

        x = self.head(x4)

        x = x.squeeze()

        return x
