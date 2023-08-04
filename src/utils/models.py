from torch import load, nn, from_numpy, no_grad, device
from rasterio import open as ropen
import numpy as np


def load_model(model: nn.Module, model_file: str, device: device):
    model.load_state_dict(load(model_file))
    model.to(device)
    model.eval()
    return model


def predict_image(model: nn.Module, image: str, device: device, patch_size=64):
    # Load an image
    with ropen(image) as src:
        image = src.read([3, 2, 1, 4], out_dtype=np.float32)

    # Get the size of the image

    image_size = image.shape[1]

    # Prepare the result array
    result = np.zeros((image_size, image_size))

    # Iterate over the image patches
    for i in range(0, image_size, patch_size):
        for j in range(0, image_size, patch_size):
            # Extract the patch
            patch = image[:, i : i + patch_size, j : j + patch_size]

            # Convert the patch to a tensor, add a batch dimension and move to the correct device
            patch_tensor = from_numpy(patch).unsqueeze(0).to(device)

            # Make the prediction
            with no_grad():
                prediction = model(patch_tensor)

            # Move the prediction to CPU and store it in the result array
            result[i : i + patch_size, j : j + patch_size] = (
                prediction.squeeze().cpu().numpy()
            )

    return result
