from segmentation_models_pytorch.decoders.unet.decoder import DecoderBlock, UnetDecoder
from torch import Tensor

from . import Unet


class ResidualDecoderBlock(DecoderBlock):
    def __init__(self, *args, **kwargs):
        super(ResidualDecoderBlock, self).__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return super(ResidualDecoderBlock, self).forward(x) + x


class ResidualUnet(Unet):
    def __init__(self, *args, **kwargs):
        super(ResidualUnet, self).__init__(*args, **kwargs)
        self.name = f"residual-{self.name}"

    def _initialize_decoder(self, *args, **kwargs):
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=self.decoder_channels,
            n_blocks=self.decoder_n_blocks,
            use_batchnorm=self.decoder_use_batchnorm,
            center=self.decoder_center,
            attention_type=self.decoder_attention_type,
            block=ResidualDecoderBlock,
        )
