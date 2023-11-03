from typing import Any, Dict, List, Optional, Union

from segmentation_models_pytorch import Unet as SegmentationUnet
from torch import Tensor

from .decoder import UnetDecoder


class Unet(SegmentationUnet):
    def __init__(
        self,
        encoder_name: str = "efficientnet-b2",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = None,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        decoder_attention_type: Optional[str] = "scse",
        in_channels: int = 5,
        classes: int = 1,
        activation: Union[str, Any, None] = None,
        aux_params: Optional[Dict] = None,
        **kwargs: Any
    ):
        super(Unet, self).__init__(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            aux_params=aux_params,
            **kwargs
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.initialize()

    def count_params(self) -> int:
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def forward(self, x: Tensor) -> Tensor:
        x = super(Unet, self).forward(x)
        x = x.squeeze()
        return x
