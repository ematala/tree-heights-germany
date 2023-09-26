from typing import List

from torch import Tensor
from torch.nn import Module
from transformers import ViTConfig, ViTModel


class VitEncoder(Module):
    def __init__(self, config: ViTConfig):
        super(VitEncoder, self).__init__()

        self.encoder = ViTModel(config)

    def forward(self, x: Tensor) -> List[Tensor]:
        # extract hidden states from the 3rd, 6th, 9th and 12th layers
        idxs = [2, 5, 8, 11]
        # pass input through the encoder
        x = self.encoder(x)
        # extract hidden states from the specified layers and remove class token
        return [x.hidden_states[i][:, 1:, :] for i in idxs]
