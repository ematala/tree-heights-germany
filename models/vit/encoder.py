from typing import List

import timm

from .backbone import _make_vit_b16_backbone, _make_vit_b_rn50_backbone


def _make_pretrained_vitb_rn50_384(
    hooks: List[int],
    readout_op: str = "ignore",
    enable_attention_hooks: bool = False,
    use_vit_only: bool = False,
):
    model = timm.create_model("vit_base_resnet50_384")

    return _make_vit_b_rn50_backbone(
        model,
        features=[256, 512, 768, 768],
        hooks=hooks,
        use_vit_only=use_vit_only,
        readout_op=readout_op,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_vitl16_384(
    hooks: List[int],
    readout_op: str = "ignore",
    enable_attention_hooks: bool = False,
):
    model = timm.create_model("vit_large_patch16_384")

    return _make_vit_b16_backbone(
        model,
        features=[256, 512, 1024, 1024],
        hooks=hooks,
        vit_features=1024,
        readout_op=readout_op,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_vitb16_384(
    hooks: List[int],
    readout_op: str = "ignore",
    enable_attention_hooks: bool = False,
):
    model = timm.create_model("vit_base_patch16_384")

    return _make_vit_b16_backbone(
        model,
        features=[96, 192, 384, 768],
        hooks=hooks,
        readout_op=readout_op,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_deitb16_384(
    hooks: List[int],
    readout_op: str = "ignore",
    enable_attention_hooks: bool = False,
):
    model = timm.create_model("deit_base_patch16_384")
    return _make_vit_b16_backbone(
        model,
        features=[96, 192, 384, 768],
        hooks=hooks,
        readout_op=readout_op,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_deitb16_384t(
    hooks: List[int],
    readout_op: str = "ignore",
    enable_attention_hooks: bool = False,
):
    model = timm.create_model("deit_tiny_patch16_224")
    return _make_vit_b16_backbone(
        model,
        features=[96, 192, 384, 768],
        hooks=hooks,
        readout_op=readout_op,
        enable_attention_hooks=enable_attention_hooks,
    )


def make_encoder(
    backbone: str,
    readout_op: str = "ignore",
    enable_attention_hooks: bool = False,
):
    return {
        "vitb16_384": _make_pretrained_vitb16_384(
            readout_op=readout_op,
            hooks=[2, 5, 8, 11],
            enable_attention_hooks=enable_attention_hooks,
        ),
        "vitl16_384": _make_pretrained_vitl16_384(
            readout_op=readout_op,
            hooks=[5, 11, 17, 23],
            enable_attention_hooks=enable_attention_hooks,
        ),
        "vit_deit_base_patch16_384": _make_pretrained_deitb16_384(
            hooks=[2, 5, 8, 11],
            readout_op=readout_op,
            enable_attention_hooks=enable_attention_hooks,
        ),
        # todo add tiny deit models
        "vitb_rn50_384": _make_pretrained_vitb_rn50_384(
            hooks=[0, 1, 8, 11],
            readout_op=readout_op,
            enable_attention_hooks=enable_attention_hooks,
        ),
    }[backbone]
