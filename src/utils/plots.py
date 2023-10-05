import os

from geopandas import read_file, sjoin
from matplotlib.pyplot import figure, show, subplots
from numpy import minimum, ndarray
from rasterio import open as ropen
from rasterio.plot import show as rshow

from .misc import get_normalized_image
from .preprocessing import Preprocessor


def brighten(image: ndarray, factor: float = 5.0) -> ndarray:
    return minimum(image * factor, 1)


def plot_image_and_prediction(image: ndarray, prediction: ndarray):
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
    image = brighten(image)

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


def plot_image_channels(image: str) -> None:
    with ropen(os.path.join(image)) as src:
        img = get_normalized_image(src)

        red, green, blue, nir, ndvi = img

        rgb, red, green, blue = [brighten(c) for c in [img[:3], red, green, blue]]

        _, (axrgb, axr, axg, axb, axnir, axndvi) = subplots(1, 6, figsize=(31, 7))

        rshow(rgb, ax=axrgb, title="RGB", vmin=0, vmax=1)
        rshow(red, ax=axr, cmap="Reds", title="Red", vmin=0, vmax=1)
        rshow(green, ax=axg, cmap="Greens", title="Green", vmin=0, vmax=1)
        rshow(blue, ax=axb, cmap="Blues", title="Blue", vmin=0, vmax=1)
        rshow(nir, ax=axnir, cmap="viridis", title="NIR", vmin=0, vmax=1)
        rshow(ndvi, ax=axndvi, cmap="RdYlGn", title="NDVI", vmin=0, vmax=1)

        for ax in [axr, axg, axb, axnir, axndvi, axrgb]:
            ax.axis("off")

        show()


def plot_labels_in_germany(shapefile: str = "data/germany/germany.geojson") -> None:
    preprocessor = Preprocessor()
    preprocessor._load_gedi()

    gedi = preprocessor.gedi

    germany = read_file(os.path.join(shapefile)).to_crs("EPSG:3857")

    gedi = gedi[(gedi.rh98 > 3) & (gedi.rh98 < 50)]

    gedi_germany = sjoin(gedi, germany, how="inner", op="within")

    fig, ax = subplots(figsize=(10, 10))

    gedi_germany.plot(
        cmap="viridis",
        column="rh98",
        marker="x",
        markersize=0.1,
        ax=ax,
    )
    ax.set_aspect("equal", "box")
    ax.set_axis_off()

    show()
