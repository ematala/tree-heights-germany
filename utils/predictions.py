import os
import re
from functools import partial
from itertools import chain
from multiprocessing import get_context
from typing import Dict, Tuple

from numpy import float32, ndarray, zeros
from PIL.Image import fromarray
from rasterio import open as ropen
from torch import Tensor, from_numpy, no_grad
from torch import device as Device
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from .misc import get_normalized_image, get_window_bounds
from .models import load


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
    image = image.squeeze().cpu().numpy()
    outputs = outputs.squeeze().cpu().numpy()

    return image, outputs


def predict_image(
    model: Module, device: Device, img: str, patch_size: int
) -> Tuple[ndarray, ndarray]:
    with ropen(os.path.join(img)) as src:
        image = get_normalized_image(src)

    # Initialize an array to hold the predictions
    _, height, width = image.shape
    outputs = zeros((1, height, width))

    # Calculate the number of patches
    n_patches = (height // patch_size) ** 2

    # Iterate through the patches
    for patch in tqdm(range(n_patches)):
        # Get the window bounds for the patch
        bounds = get_window_bounds(patch, patch_size)
        row_start, row_end, col_start, col_end = bounds

        # Extract the patch
        patch = (
            from_numpy(image[:, row_start:row_end, col_start:col_end].astype(float32)),
            None,
        )

        # Use the predict_patch function to get the prediction
        _, prediction = predict_patch(model, patch, device)

        # Place the patch prediction into the prediction image
        outputs[:, row_start:row_end, col_start:col_end] = prediction

    return image, outputs.squeeze()


def save_prediction(prediction: ndarray, filename: str) -> None:
    fromarray(prediction, mode="F").save(f"{filename}", "TIFF")


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


def predict_all_images(
    img_dir: str,
    patch_size: int,
    model_path: str,
    device: Device,
    filename: str,
):
    regexp = r"L15\-\d{4}E\-\d{4}N\.tif"
    images = [f for f in os.listdir(img_dir) if re.match(regexp, f)]

    model = load(model_path)
    fn = partial(model)
    desc = "Predicting images"

    with get_context("spawn").Pool(os.cpu_count() // 2) as pool:
        res = [r for r in tqdm(pool.imap_unordered(fn, images), desc, len(images))]

    return list(chain.from_iterable(res))
