import types

import timm
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from .backbone import (
    _forward_flex,
    get_postprocessing_layers,
    get_unflatten_layer,
    _resize_pos_embed,
)
from .hooks import get_activation, get_attention


def make_encoder(
    backbone: str,
    readout_op: str = "ignore",
    enable_attention_hooks: bool = False,
):
    config = {
        "img_size": 256,
        "in_chans": 5,
        "patch_size": 16,
        "num_heads": 8,
        "embed_dim": 128,
    }
    return {
        "vitb16_256": VitEncoder(
            model=timm.create_model("vit_base_patch16_384", **config),
            vit_features=128,
            features=[64, 96, 128, 128],
            hooks=[2, 5, 8, 11],
            readout_op=readout_op,
            enable_attention_hooks=enable_attention_hooks,
        ),
    }[backbone]


class VitEncoder(nn.Module):
    def __init__(
        self,
        model: Module,
        vit_features=128,
        features=[64, 96, 128, 128],
        hooks=[2, 5, 8, 11],
        img_size=[256, 256],
        patch_size=[16, 16],
        readout_op="ignore",
        start_index=1,
        enable_attention_hooks=False,
    ) -> None:
        super(VitEncoder, self).__init__()

        self.model = model
        self.model.start_index = start_index
        self.model.patch_size = patch_size

        self.activations = {}
        self.attention = {}

        for i, hook in enumerate(hooks):
            self.model.blocks[hook].register_forward_hook(
                get_activation(str(i + 1), self.activations)
            )

        if enable_attention_hooks:
            self.attention = {}
            for i, hook in enumerate(hooks):
                self.model.blocks[hook].attn.register_forward_hook(
                    get_attention(str(i + 1), self.attention)
                )

        self.unflatten = get_unflatten_layer(img_size, patch_size)

        self.postprocessing = get_postprocessing_layers(
            img_size, patch_size, vit_features, features, readout_op, start_index
        )

        self.model._resize_pos_embed = types.MethodType(_resize_pos_embed, self.model)
        self.model.forward_flex = types.MethodType(_forward_flex, self.model)

    def forward(self, x: Tensor):
        _ = self.model.forward_flex(x)

        layer_1 = self.activations["1"]
        layer_2 = self.activations["2"]
        layer_3 = self.activations["3"]
        layer_4 = self.activations["4"]

        layer_1 = self.postprocessing[0][0:2](layer_1)
        layer_2 = self.postprocessing[1][0:2](layer_2)
        layer_3 = self.postprocessing[2][0:2](layer_3)
        layer_4 = self.postprocessing[3][0:2](layer_4)

        if layer_1.ndim == 3:
            layer_1 = self.unflatten(layer_1)
        if layer_2.ndim == 3:
            layer_2 = self.unflatten(layer_2)
        if layer_3.ndim == 3:
            layer_3 = self.unflatten(layer_3)
        if layer_4.ndim == 3:
            layer_4 = self.unflatten(layer_4)

        layer_1 = self.postprocessing[0][3 : len(self.postprocessing[0])](layer_1)
        layer_2 = self.postprocessing[1][3 : len(self.postprocessing[1])](layer_2)
        layer_3 = self.postprocessing[2][3 : len(self.postprocessing[2])](layer_3)
        layer_4 = self.postprocessing[3][3 : len(self.postprocessing[3])](layer_4)

        return layer_1, layer_2, layer_3, layer_4
