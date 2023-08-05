from matplotlib.pyplot import show, subplots
from numpy import ndarray


def plot_image_and_prediction(image: ndarray, prediction: ndarray):
    # Transpose the image and prediction to have the channels last
    image = image[:3, :, :].transpose(1, 2, 0)

    # Create a subplot to display the image and prediction side by side
    _, (ax1, ax2) = subplots(1, 2, figsize=(12, 6))

    # Plot the original image
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Plot the prediction
    ax2.imshow(prediction.squeeze(), cmap="viridis")
    ax2.set_title("Predicted Height Map")
    ax2.axis("off")

    show()
