import os
from typing import Dict, Optional, Tuple

import rasterio
from geopandas import GeoDataFrame, read_file, sjoin
from matplotlib import pyplot as plt
from numpy import arange, expand_dims, minimum, ndarray
from rasterio.plot import show
from torch import Tensor

from models.vit.hooks import get_mean_attention_map

from .io import read_window
from .transforms import add_ndvi, denormalize, scale

CONFIG = {
    "format": "pdf",
    "dpi": 300,
    "bbox_inches": "tight",
    "pad_inches": 0.1,
    "transparent": True,
}


def save_or_show_plot(path: Optional[str] = None) -> None:
    """Save or show the current plot.

    Args:
        path (str, optional): Path to save the plot. Defaults to None.
    """
    plt.savefig(path, **CONFIG) if path else plt.show()


def brighten_image(image: ndarray, factor: float = 5.0) -> ndarray:
    return minimum(image * factor, 1)


def plot_image_and_prediction(
    image: ndarray,
    prediction: ndarray,
    path: Optional[str] = None,
):
    """Plots the original image and the prediction side by side.

    Args:
        image (ndarray): The original RGB image.
        prediction (ndarray): The predicted height map.
    """
    # Transpose the image and prediction to have channels last
    image = image[:3, :, :].transpose(1, 2, 0)

    # Brighten the image using the simplified method
    image = brighten_image(image)

    # Create a custom grid for the image, prediction, and colorbar
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_axes([0.05, 0.1, 0.45, 0.8])
    ax2 = fig.add_axes([0.51, 0.1, 0.45, 0.8])
    cax = fig.add_axes([0.97, 0.1, 0.01, 0.8])

    # Plot the original image
    ax1.imshow(image)
    ax1.set_title("RGB Image")
    ax1.axis("off")

    # Plot the prediction
    im = ax2.imshow(prediction.squeeze(), cmap="inferno")
    ax2.set_title("Predicted Height Map")
    ax2.axis("off")

    # Add colorbar to the dedicated axis
    fig.colorbar(im, cax=cax, orientation="vertical").set_label(
        "Height (meters)", rotation=-90, va="bottom"
    )

    save_or_show_plot(path)


def plot_image_channels(
    image: str,
    path: Optional[str] = None,
) -> None:
    with rasterio.open(os.path.join(image)) as src:
        img = src.read([3, 2, 1, 4])
        img = add_ndvi(img)
        img[:4] = scale(img[:4])

        red, green, blue, nir, ndvi = img

        rgb, red, green, blue = [brighten_image(c) for c in [img[:3], red, green, blue]]

        _, (axrgb, axr, axg, axb, axnir, axndvi) = plt.subplots(1, 6, figsize=(31, 7))

        show(rgb, ax=axrgb, title="RGB", vmin=0, vmax=1)
        show(red, ax=axr, cmap="Reds", title="Red", vmin=0, vmax=1)
        show(green, ax=axg, cmap="Greens", title="Green", vmin=0, vmax=1)
        show(blue, ax=axb, cmap="Blues", title="Blue", vmin=0, vmax=1)
        show(nir, ax=axnir, cmap="inferno", title="NIR", vmin=0, vmax=1)
        show(ndvi, ax=axndvi, cmap="RdYlGn", title="NDVI", vmin=0, vmax=1)

        for ax in [axr, axg, axb, axnir, axndvi, axrgb]:
            ax.axis("off")

        save_or_show_plot(path)


def plot_labels_in_germany(
    gedi: GeoDataFrame,
    shapefile: str = "data/germany/germany.geojson",
    path: Optional[str] = None,
) -> None:
    germany = read_file(os.path.join(shapefile)).to_crs("EPSG:3857")

    gedi_germany = sjoin(gedi, germany, how="inner", op="within")

    fig, ax = plt.subplots(figsize=(10, 10))

    gedi_germany.plot(
        cmap="inferno",
        column="rh98",
        marker="x",
        markersize=0.1,
        ax=ax,
    )
    ax.set_aspect("equal", "box")
    ax.set_axis_off()

    save_or_show_plot(path)


def plot_predictions(
    images: Tensor,
    predictions: dict,
    path: Optional[str] = None,
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
        image = denormalize(images[i])
        image = scale(image)
        image = image[:3, :, :].permute(1, 2, 0).numpy()

        image = brighten_image(image)

        axes[i, 0].imshow(image)
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title("RGB Image")

        col = 1
        for model_name, preds in predictions.items():
            pred = preds[i].squeeze().numpy()
            im = axes[i, col].imshow(pred, cmap="inferno")
            axes[i, col].axis("off")
            if i == 0:
                axes[i, col].set_title(model_name)
            col += 1

    cax = fig.add_axes([0.88, 0.1, 0.01, 0.8])
    fig.colorbar(im, cax=cax, orientation="vertical").set_label(
        "Height (meters)", rotation=-90, va="bottom"
    )

    save_or_show_plot(path)


def plot_true_vs_predicted_scatter(
    true_values: ndarray,
    predicted_values: ndarray,
    model_name: str,
    path: Optional[str] = None,
    bounds: Optional[Tuple[int, int]] = (0, 50),
) -> None:
    """
    Create a scatter plot for true vs predicted values with specific adjustments.

    Args:
        true_values (np.ndarray): Ground truth values
        predicted_values (np.ndarray): Model's predictions
        model_name (str): Name of the model

    Raises:
        ValueError: If the shapes of true_values and predicted_values do not match.
    """
    if true_values.shape != predicted_values.shape:
        raise ValueError("The shapes of true_values and predicted_values must match.")

    plt.figure(figsize=(10, 10))

    # Only plot values within the bounds
    lower, upper = bounds
    mask = (true_values >= lower) & (true_values < upper)
    true_values, predicted_values = true_values[mask], predicted_values[mask]

    plt.scatter(true_values, predicted_values, marker="o", alpha=0.6)

    # 1:1 line for perfect predictions
    min_val = min(true_values.min(), predicted_values.min())
    max_val = max(true_values.max(), predicted_values.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

    plt.xlabel("GEDI RH98 (m)", fontsize=14)
    plt.ylabel(f"{model_name} RH98 (m)", fontsize=14)

    ticks = arange(0, 55, 5)
    plt.xticks(ticks)
    plt.yticks(ticks)

    save_or_show_plot(path)


def plot_true_vs_predicted_histogram(
    true_labels: ndarray,
    predicted_values: ndarray,
    model_name: str,
    path: Optional[str] = None,
    bins=range(50),
) -> None:
    """
    Plot a histogram of the true and predicted values.

    Args:
        true_labels (ndarray): Ground truth values.
        predicted_values (ndarray): Model's predictions.
        model_name (str): Name of the model.
        bins (_type_, optional): Bins for the histogram. Defaults to range(50).
    """
    range_tuples = [(i, i + 5) for i in list(range(0, 50, 5))]
    positions = [(start + end) / 2 for start, end in range_tuples]
    errors = predicted_values - true_labels
    boxes = [
        errors[(true_labels >= start) & (true_labels < end)]
        for start, end in range_tuples
    ]

    fig, ax = plt.subplots(figsize=(16, 9))
    ax2 = ax.twinx()

    ax.hist(
        true_labels,
        bins=bins,
        color="lightgrey",
        label="GEDI RH98",
    )
    ax.hist(
        predicted_values,
        bins=bins,
        histtype="step",
        edgecolor="black",
        label=f"{model_name}",
        linewidth=1,
    )

    ax2.boxplot(
        boxes,
        positions=positions,
        vert=True,
        widths=0.5,
        whis=[5, 95],
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="black", edgecolor="black"),
        medianprops=dict(color="lightgrey"),
    )

    ax2.axhline(y=0, color="grey", linestyle="--")

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{start}-{end}m" for start, end in range_tuples], fontsize=10)
    ax.set_xlabel("Height (m)", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax2.set_ylabel("Error (m)", fontsize=14)
    ax.legend(loc="upper right")

    plt.tight_layout()

    save_or_show_plot(path)


def compare_predictions(
    img: str, models: Dict[str, str] = {}, path: Optional[str] = None
):
    with rasterio.open(img) as src:
        image = src.read([3, 2, 1])
        image = scale(image)
        image = brighten_image(image)

        num_plots = 1 + len(models)

        _, axs = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

        if num_plots == 1:
            axs = [axs]

        show(image, ax=axs[0], title=img)

        for i, (name, fp) in enumerate(models.items()):
            pred = read_window(fp, src.meta, src.bounds)
            show(pred, ax=axs[i + 1], title=f"Predictions from {name}", cmap="inferno")

        for ax in axs:
            ax.axis("off")

    save_or_show_plot(path)


def visualize_attention(input, model, prediction, path: Optional[str] = None):
    input = (input + 1.0) / 2.0

    attn1 = model.encoder.attention["1"]
    attn2 = model.encoder.attention["2"]
    attn3 = model.encoder.attention["3"]
    attn4 = model.encoder.attention["4"]

    plt.subplot(3, 4, 1), plt.imshow(input.squeeze().permute(1, 2, 0)), plt.title(
        "Input", fontsize=8
    ), plt.axis("off")
    plt.subplot(3, 4, 2), plt.imshow(prediction), plt.set_cmap("inferno"), plt.title(
        "Prediction", fontsize=8
    ), plt.axis("off")

    h = [2, 5, 8, 11]

    # upper left
    plt.subplot(345)
    plt.imshow(get_mean_attention_map(attn1, 1, input.shape))
    plt.ylabel("Upper left corner", fontsize=8)
    plt.title(f"Layer {h[0] + 1}", fontsize=8)
    gc = plt.gca()
    gc.axes.xaxis.set_ticklabels([])
    gc.axes.yaxis.set_ticklabels([])
    gc.axes.xaxis.set_ticks([])
    gc.axes.yaxis.set_ticks([])

    plt.subplot(346)
    plt.imshow(get_mean_attention_map(attn2, 1, input.shape))
    plt.title(f"Layer {h[1] + 1}", fontsize=8)
    plt.axis("off")

    plt.subplot(347)
    plt.imshow(get_mean_attention_map(attn3, 1, input.shape))
    plt.title(f"Layer {h[2] + 1}", fontsize=8)
    plt.axis("off")

    plt.subplot(348)
    plt.imshow(get_mean_attention_map(attn4, 1, input.shape))
    plt.title(f"Layer {h[3] + 1}", fontsize=8)
    plt.axis("off")

    # lower right
    plt.subplot(3, 4, 9), plt.imshow(get_mean_attention_map(attn1, -1, input.shape))
    plt.ylabel("Lower right corner", fontsize=8)
    gc = plt.gca()
    gc.axes.xaxis.set_ticklabels([])
    gc.axes.yaxis.set_ticklabels([])
    gc.axes.xaxis.set_ticks([])
    gc.axes.yaxis.set_ticks([])

    plt.subplot(3, 4, 10)
    plt.imshow(get_mean_attention_map(attn2, -1, input.shape))
    plt.axis("off")
    plt.subplot(3, 4, 11)
    plt.imshow(get_mean_attention_map(attn3, -1, input.shape)), plt.axis("off")
    plt.subplot(3, 4, 12)
    plt.imshow(get_mean_attention_map(attn4, -1, input.shape)), plt.axis("off")
    plt.tight_layout()

    save_or_show_plot(path)
