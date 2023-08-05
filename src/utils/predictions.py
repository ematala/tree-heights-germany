from typing import Tuple

from numpy import ndarray
from torch import Tensor, no_grad
from torch import device as Device
from torch.nn import Module


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
