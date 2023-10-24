import os

from geopandas import read_file, sjoin
from matplotlib import pyplot as plt
from numpy import arange, expand_dims, minimum, ndarray
from rasterio import open as ropen
from rasterio.plot import show as rshow
from torch import Tensor

from .misc import get_normalized_image
from .preprocessing import Preprocessor


def brighten(image: ndarray, factor: float = 5.0) -> ndarray:
    return minimum(image * factor, 1)


def plot_image_and_prediction(image: ndarray, prediction: ndarray):
    """Plots the original image and the prediction side by side.

    Args:
        image (ndarray): The original RGB image.
        prediction (ndarray): The predicted height map.
    """
    # Transpose the image and prediction to have channels last
    image = image[:3, :, :].transpose(1, 2, 0)

    # Brighten the image using the simplified method
    image = brighten(image)

    # Create a custom grid for the image, prediction, and colorbar
    fig = plt.figure(figsize=(14, 6))
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

    plt.show()


def plot_image_channels(image: str) -> None:
    with ropen(os.path.join(image)) as src:
        img = get_normalized_image(src)

        red, green, blue, nir, ndvi = img

        rgb, red, green, blue = [brighten(c) for c in [img[:3], red, green, blue]]

        _, (axrgb, axr, axg, axb, axnir, axndvi) = plt.subplots(1, 6, figsize=(31, 7))

        rshow(rgb, ax=axrgb, title="RGB", vmin=0, vmax=1)
        rshow(red, ax=axr, cmap="Reds", title="Red", vmin=0, vmax=1)
        rshow(green, ax=axg, cmap="Greens", title="Green", vmin=0, vmax=1)
        rshow(blue, ax=axb, cmap="Blues", title="Blue", vmin=0, vmax=1)
        rshow(nir, ax=axnir, cmap="viridis", title="NIR", vmin=0, vmax=1)
        rshow(ndvi, ax=axndvi, cmap="RdYlGn", title="NDVI", vmin=0, vmax=1)

        for ax in [axr, axg, axb, axnir, axndvi, axrgb]:
            ax.axis("off")

        plt.show()


def plot_labels_in_germany(shapefile: str = "data/germany/germany.geojson") -> None:
    preprocessor = Preprocessor()
    preprocessor._load_gedi()

    gedi = preprocessor.gedi

    germany = read_file(os.path.join(shapefile)).to_crs("EPSG:3857")

    gedi = gedi[(gedi.rh98 > 3) & (gedi.rh98 < 50)]

    gedi_germany = sjoin(gedi, germany, how="inner", op="within")

    fig, ax = plt.subplots(figsize=(10, 10))

    gedi_germany.plot(
        cmap="viridis",
        column="rh98",
        marker="x",
        markersize=0.1,
        ax=ax,
    )
    ax.set_aspect("equal", "box")
    ax.set_axis_off()

    plt.show()


def plot_predictions(
    images: Tensor,
    predictions: dict,
    nrows: int = 4,
) -> None:
    """Plot predictions for a set of images.

    Args:
        images (Tensor): The original RGB images.
        predictions (dict): The predicted height maps.
        nrows (int, optional): Number of rows to plot. Defaults to 4.

    Raises:
        ValueError: If the number of rows requested is greater than the number of samples available.
    """
    total_rows = images.shape[0]
    if total_rows < nrows:
        raise ValueError(
            f"Requested {nrows} rows, but only {total_rows} samples available."
        )

    for model_name, preds in predictions.items():
        if total_rows != preds.shape[0]:
            raise ValueError(
                f"Dimension mismatch: {total_rows} original images but {preds.shape[0]} predictions for {model_name}"
            )

    ncols = len(predictions) + 1
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 4 * nrows),
    )

    fig.subplots_adjust(right=0.85, wspace=0.1, hspace=0.1)

    if nrows == 1:
        axes = expand_dims(axes, axis=0)

    for i in range(nrows):
        image = images[i, :3, :, :].permute(1, 2, 0).numpy()

        image = brighten(image)

        axes[i, 0].imshow(image)
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title("RGB Image")

        col = 1
        for model_name, preds in predictions.items():
            pred = preds[i].squeeze().numpy()
            im = axes[i, col].imshow(pred, cmap="viridis")
            axes[i, col].axis("off")
            if i == 0:
                axes[i, col].set_title(model_name)
            col += 1

    cax = fig.add_axes([0.88, 0.1, 0.01, 0.8])
    fig.colorbar(im, cax=cax, orientation="vertical").set_label(
        "Height (meters)", rotation=-90, va="bottom"
    )

    plt.show()


def plot_true_vs_predicted(
    true_values: ndarray,
    predicted_values: ndarray,
    model_name: str,
) -> None:
    """
    Create a scatter plot for true vs predicted values with specific adjustments.

    Parameters:
    - true_values (np.ndarray): Ground truth values
    - predicted_values (np.ndarray): Model's predictions

    Raises:
    - ValueError: If the shapes of true_values and predicted_values do not match.
    """
    if true_values.shape != predicted_values.shape:
        raise ValueError("The shapes of true_values and predicted_values must match.")

    plt.figure(figsize=(10, 10))

    # Scatter plot
    plt.scatter(true_values, predicted_values, marker="o", edgecolor="k", alpha=0.7)

    # 1:1 line for perfect predictions
    min_val = min(true_values.min(), predicted_values.min())
    max_val = max(true_values.max(), predicted_values.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

    plt.xlabel("GEDI RH98 (m)", fontsize=14)
    plt.ylabel(f"{model_name} RH98 (m)", fontsize=14)

    ticks = arange(0, max_val, 5)
    plt.xticks(ticks)
    plt.yticks(ticks)

    plt.show()
