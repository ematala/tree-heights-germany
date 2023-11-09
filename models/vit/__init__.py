import torch
import torch.nn as nn

from .blocks import (
    FeatureFusionBlock_custom,
    Interpolate,
)
from .decoder import make_decoder
from .encoder import forward_vit, make_encoder


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class Vit(nn.Module):
    def __init__(
        self,
        features=128,
        backbone="vitb16_384",
        readout="ignore",
        channels_last=False,
        use_bn=True,
        enable_attention_hooks=False,
        non_negative=True,
    ):
        super(Vit, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        self.encoder = make_encoder(
            backbone,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.decoder = make_decoder(backbone, features)

        self.refinenet1 = _make_fusion_block(features, use_bn)
        self.refinenet2 = _make_fusion_block(features, use_bn)
        self.refinenet3 = _make_fusion_block(features, use_bn)
        self.refinenet4 = _make_fusion_block(features, use_bn)

        self.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

    def forward(self, x):
        if self.channels_last:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.encoder, x)

        layer_1_rn = self.decoder.layer1_rn(layer_1)
        layer_2_rn = self.decoder.layer2_rn(layer_2)
        layer_3_rn = self.decoder.layer3_rn(layer_3)
        layer_4_rn = self.decoder.layer4_rn(layer_4)

        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)

        out = self.output_conv(path_1)

        return out.squeeze(dim=1)
