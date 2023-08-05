import os
from typing import Tuple

from numpy import float32, ndarray, zeros
from rasterio import open as ropen
from torch import Tensor, from_numpy, no_grad
from torch import device as Device
from torch.nn import Module

from src.utils.misc import get_window_bounds


def predict_patch(
    model: Module, patch: Tuple[Tensor, Tensor], device: Device
) -> Tuple[ndarray, ndarray]:
    model.eval()

    # Extract the image from the patch
    image, _ = patch

    # Add batch dimension and move to the device
    image = image.unsqueeze(0).to(device)

    # Perform the prediction
    with no_grad():
        outputs = model(image)

    # Move the image and prediction to CPU and remove batch dimension
    image = image.squeeze(0).cpu().numpy()
    outputs = outputs.squeeze(0).cpu().numpy()

    return image, outputs


def predict_image(
    model: Module, device: Device, img: str, patch_size: int = 64
) -> Tuple[ndarray, ndarray]:
    with ropen(os.path.join(img)) as src:
        image = src.read([3, 2, 1, 4])
        image = (image - image.min()) / (image.max() - image.min())

    # Initialize an array to hold the predictions
    _, height, width = image.shape
    outputs = zeros((1, height, width))

    # Calculate the number of patches
    size = patch_size
    n_patches = (height // size) * (width // size)

    # Iterate through the patches
    for patch in range(n_patches):
        # Get the window bounds for the patch
        row, col = get_window_bounds(patch)

        # Extract the patch
        patch = (
            from_numpy(image[:, row : row + size, col : col + size].astype(float32)),
            None,
        )

        # Use the predict_patch function to get the prediction
        _, prediction = predict_patch(model, patch, device)

        # Place the patch prediction into the prediction image
        outputs[:, row : row + size, col : col + size] = prediction

    return image, outputs.squeeze()
