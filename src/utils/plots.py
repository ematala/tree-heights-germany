import os
import matplotlib.pyplot as plt
import numpy as np
from rasterio import open as ropen


def plot_sample(image: str, outputs: np.ndarray) -> None:
    # Create numpy arrays from tensors
    image = image.numpy()
    outputs = outputs.detach().numpy()

    # Normalize RGB values
    image = image / image.max()

    # Swap red and blue bands
    image[[0, 2]] = image[[2, 0]]

    # Remove NIR band
    image = image[:3]

    # Transform from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))

    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Show image
    axs[0].imshow(image)
    axs[0].set_title(image)
    axs[0].axis("off")

    # Show model outputs
    axs[1].imshow(outputs.squeeze(), cmap="viridis")
    axs[1].set_title("Predicted labels")
    axs[1].axis("off")

    # Show figure
    plt.show()


def plot_image(image: str) -> None:
    with ropen(os.path.join(image)) as src:
        img = src.read([3, 2, 1])

    img = img / img.max()
    img = np.transpose(img, (1, 2, 0))

    plt.imshow(img)
    plt.axis("off")
    plt.title(image)
    plt.show()


def plot_prediction(prediction: np.ndarray) -> None:
    plt.imshow(prediction.squeeze(), cmap="viridis")
    plt.axis("off")
    plt.title("Predicted labels")
    plt.show()
