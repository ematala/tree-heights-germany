from torch import nn
from transformers import ViTFeatureExtractor, ViTModel


class ViTEncoder(nn.Module):
    def __init__(self, pretrained_model_name):
        super(ViTEncoder, self).__init__()
        # Instantiate a feature extractor and a model with the given pre-trained model name
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            pretrained_model_name
        )
        self.model = ViTModel.from_pretrained(pretrained_model_name)

    def forward(self, images):
        # Extract features from the images using the feature extractor.
        # The resulting tensor is then passed to the model
        inputs = self.feature_extractor(
            images=images, return_tensors="pt", padding="max_length", truncation=True
        )
        outputs = self.model(**inputs)
        # We return the last hidden state which contains the feature representations of the input image patches
        return outputs.last_hidden_state


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class UNetDecoder(nn.Module):
    def __init__(self, vit_channels, hidden_channels, output_channels):
        super().__init__()

        self.upconv1 = nn.ConvTranspose2d(
            vit_channels, hidden_channels, kernel_size=2, stride=2
        )
        self.conv1 = ConvBlock(hidden_channels, hidden_channels)
        self.upconv2 = nn.ConvTranspose2d(
            hidden_channels, hidden_channels // 2, kernel_size=2, stride=2
        )
        self.conv2 = ConvBlock(hidden_channels // 2, hidden_channels // 2)
        self.final_conv = nn.Conv2d(
            hidden_channels // 2, output_channels, kernel_size=1
        )

    def forward(self, x):
        x = self.upconv1(x)
        x = self.conv1(x)
        x = self.upconv2(x)
        x = self.conv2(x)
        x = self.final_conv(x)

        return x
