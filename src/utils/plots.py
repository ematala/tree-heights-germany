from matplotlib.pyplot import figure, show
from numpy import minimum, ndarray


def plot_image_and_prediction(
    image: ndarray, prediction: ndarray, brighten: float = 1.0
):
    """
    Plots the original image and the prediction side by side.

    Parameters:
        - image: The original RGB image.
        - prediction: The predicted height map.
        - brighten: A factor to brighten the image. Default is 1.0 (no change).
    """

    # Transpose the image and prediction to have channels last
    image = image[:3, :, :].transpose(1, 2, 0)

    # Brighten the image using the simplified method
    image = minimum(image * brighten, 1)

    # Create a custom grid for the image, prediction, and colorbar
    fig = figure(figsize=(14, 6))
    ax1 = fig.add_axes([0.05, 0.1, 0.45, 0.8])
    ax2 = fig.add_axes([0.51, 0.1, 0.45, 0.8])
    cax = fig.add_axes([0.97, 0.1, 0.01, 0.8])

    # Plot the original image
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Plot the prediction
    im = ax2.imshow(prediction.squeeze(), cmap="viridis")
    ax2.set_title("Predicted Height Map")
    ax2.axis("off")

    # Add colorbar to the dedicated axis
    fig.colorbar(im, cax=cax, orientation="vertical").set_label(
        "Height (meters)", rotation=-90, va="bottom"
    )

    show()
