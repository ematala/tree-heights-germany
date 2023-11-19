from numpy import ndarray, uint8
from rasterio import open as ropen


def save_prediction(
    pred: ndarray,
    input_file: str,
    output_file: str,
    threshold: float = 5,
):
    with ropen(input_file) as src:
        meta = src.meta.copy()

    meta.update({"count": 1, "dtype": "uint8", "nodata": 0, "compress": "LZW"})

    pred[pred < threshold] = 0

    pred = (pred / pred.max() * 255).astype(uint8)

    with ropen(output_file, "w", **meta) as dst:
        dst.write(pred, 1)
