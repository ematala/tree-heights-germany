import rasterio
from numpy import ndarray, uint8


def save_prediction(
    pred: ndarray,
    input_file: str,
    output_file: str,
    threshold: float = 5,
):
    with rasterio.open(input_file) as src:
        meta = src.meta.copy()

    meta.update({"count": 1, "dtype": "uint8", "nodata": 0, "compress": "LZW"})

    pred[pred < threshold] = 0

    pred = (pred / pred.max() * 255).astype(uint8)

    with rasterio.open(output_file, "w", **meta) as dst:
        dst.write(pred, 1)


def read_window(fp: str, bounds):
    with rasterio.open(fp) as src:
        window = rasterio.windows.from_bounds(*bounds, src.transform)
        return src.read(1, window=window)
