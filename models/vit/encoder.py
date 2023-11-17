import types

import timm
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from .backbone import (
    _forward_flex,
    _resize_pos_embed,
    get_postprocessing_layers,
    get_unflatten_layer,
    make_feature_dims,
)
from .hooks import get_activation, get_attention


def make_encoder(
    backbone: str,
    embed_dim: int,
    readout_op: str = "ignore",
    enable_attention_hooks: bool = False,
    **kwargs,
):
    return {
        "vit_tiny_patch16_256": VitEncoder(
            model=timm.create_model("vit_tiny_patch16_224", **kwargs),
            embed_dim=embed_dim,
            feature_dims=make_feature_dims(embed_dim),
            hooks=[2, 5, 8, 11],
            readout_op=readout_op,
            enable_attention_hooks=enable_attention_hooks,
        ),
        "vit_small_patch16_256": VitEncoder(
            model=timm.create_model("vit_small_patch16_224", **kwargs),
            embed_dim=embed_dim,
            feature_dims=make_feature_dims(embed_dim),
            hooks=[2, 5, 8, 11],
            readout_op=readout_op,
            enable_attention_hooks=enable_attention_hooks,
        ),
        "vit_base_patch16_256": VitEncoder(
            model=timm.create_model("vit_base_patch16_224", **kwargs),
            embed_dim=embed_dim,
            feature_dims=make_feature_dims(embed_dim),
            hooks=[2, 5, 8, 11],
            readout_op=readout_op,
            enable_attention_hooks=enable_attention_hooks,
        ),
        "vit_large_patch16_256": VitEncoder(
            model=timm.create_model("vit_large_patch16_224", **kwargs),
            embed_dim=embed_dim,
            feature_dims=make_feature_dims(embed_dim),
            hooks=[5, 11, 17, 23],
            readout_op=readout_op,
            enable_attention_hooks=enable_attention_hooks,
        ),
    }[backbone]


class VitEncoder(nn.Module):
    def __init__(
        self,
        model: Module,
        embed_dim=192,
        feature_dims=[24, 48, 96, 192],
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
            img_size, patch_size, embed_dim, feature_dims, readout_op, start_index
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
