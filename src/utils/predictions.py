import os

import matplotlib.pyplot as plt
import numpy as np
from rasterio import open as ropen
from torch import device, from_numpy, nn
from torchvision.transforms.functional import to_pil_image


def extract_patches(image: np.ndarray, patch_size=64):
    """
    This function extracts patches from the given image.

    Parameters:
    - image: a numpy array representing the image.
    - patch_size: the size of the patches.

    Returns:
    - patches: a 4D numpy array containing the extracted patches.
    """
    h, w = image.shape[1:]
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size

    patches = np.empty((num_patches_h, num_patches_w, patch_size, patch_size))

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = image[
                :,
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            patches[i, j] = patch

    return patches


def predict_patches(model: nn.Module, patches: np.ndarray, device: device):
    """
    This function uses the given model to predict the output for the given patches.

    Parameters:
    - model: the trained PyTorch model.
    - patches: a 4D numpy array containing the patches.
    - device: the device to use for the computations ('cpu' or 'cuda').

    Returns:
    - predictions: a 4D numpy array containing the predicted output for each patch.
    """
    num_patches_h, num_patches_w, _, _ = patches.shape

    predictions = np.empty((num_patches_h, num_patches_w))

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = patches[i, j]

            patch_tensor = from_numpy(patch).unsqueeze(0)
            patch_tensor = patch_tensor.to(device)

            patch_output = model(patch_tensor)
            patch_output = patch_output.squeeze(0).cpu().detach().numpy()

            predictions[i, j] = patch_output

    return predictions


def predict_image(model: nn.Module, img_path: str, device: device, patch_size=64):
    model.eval()

    with ropen(os.path.join(img_path)) as src:
        image = src.read([3, 2, 1, 4], out_dtype=np.float32)

    patches = extract_patches(image, patch_size)
    predictions = predict_patches(model, patches, device)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(to_pil_image(image[:3, :, :]))
    plt.title("Input Image (RGB Channels)")

    plt.subplot(1, 2, 2)
    plt.imshow(predictions, cmap="jet")
    plt.title("Model Predictions")
    plt.show()

    return predictions
