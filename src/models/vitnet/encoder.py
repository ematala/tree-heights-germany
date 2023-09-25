from typing import Tuple

from torch import Tensor
from torch.nn import Module
from transformers import ViTConfig, ViTModel


class VitEncoder(Module):
    def __init__(self):
        super(VitEncoder, self).__init__()

        config = ViTConfig(
            image_size=256,
            patch_size=16,
            num_channels=4,
            num_hidden_layers=12,
            num_attention_heads=8,
            hidden_size=512,
            intermediate_size=2048,
            output_hidden_states=True,
        )

        self.encoder = ViTModel(config)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # extract hidden states from the 3rd, 6th, 9th and 12th layers
        idxs = [2, 5, 8, 11]
        # pass input through the encoder
        outputs = self.encoder(x)
        # extract hidden states from the specified layers and remove class token
        tb3, tb6, tb9, tb12 = [outputs.hidden_states[i][:, 1:, :] for i in idxs]

        return tb3, tb6, tb9, tb12
