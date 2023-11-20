import os
from typing import Dict, Tuple

from numpy import float32, ndarray, zeros
from rasterio import open as ropen
from torch import Tensor, from_numpy, no_grad
from torch import device as Device
from torch.nn import Module
from torch.utils.data import DataLoader

from .misc import get_normalized_image, get_window_bounds


def predict_image(
    img: str, model: Module, device: Device, patch_size: int = 256
) -> Tuple[ndarray, ndarray]:
    model.eval()

    with ropen(os.path.join(img)) as src:
        image = get_normalized_image(src)

    # Initialize an array to hold the predictions
    _, height, width = image.shape
    outputs = zeros((height, width), dtype=float32)

    # Calculate the number of patches
    n_patches = (height // patch_size) ** 2

    # Iterate through the patches
    for patch in range(n_patches):
        # Get the window bounds for the patch
        bounds = get_window_bounds(patch, patch_size)
        row_start, row_end, col_start, col_end = bounds

        # Extract the image patch
        patch = (
            from_numpy(image[:, row_start:row_end, col_start:col_end].astype(float32))
            .unsqueeze(0)
            .to(device)
        )
        # Perform the prediction
        with no_grad():
            prediction = model(patch)

        # Add the patch to the output array
        outputs[row_start:row_end, col_start:col_end] = (
            prediction.squeeze().cpu().numpy()
        )

    return image, outputs


def predict_batch(
    models: Dict[str, Module],
    loader: DataLoader,
    device: Device,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    predictions = {}

    inputs, _ = next(iter(loader))

    for model_name, model in models.items():
        model.to(device)
        model.eval()

        with no_grad():
            outputs = model(inputs.to(device))

        predictions[model_name] = outputs.cpu()

    return inputs, predictions
