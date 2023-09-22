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

    def forward(self, x):
        idxs = [2, 5, 8, 11]
        outputs = self.encoder(x)
        tb3, tb6, tb9, tb12 = [outputs.hidden_states[i] for i in idxs]

        return tb3, tb6, tb9, tb12
