import numpy as np
import rasterio
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    transform_bounds,
)
from rasterio.windows import from_bounds


def save_prediction(
    pred: np.ndarray,
    input_file: str,
    output_file: str,
    threshold: float = 5,
):
    with rasterio.open(input_file) as src:
        meta = src.meta.copy()

    meta.update({"count": 1, "dtype": "uint16", "nodata": 0, "compress": "LZW"})

    pred[pred < threshold] = 0

    pred = (pred / pred.max() * 65535).astype(np.uint16)

    with rasterio.open(output_file, "w", **meta) as dst:
        dst.write(pred, 1)


def read_window(fp: str, meta, bounds):
    with rasterio.open(fp) as src:
        if src.crs == meta.get("crs"):
            return src.read(1, window=from_bounds(*bounds, src.transform))

        src_bounds = transform_bounds(meta.get("crs"), src.crs, *bounds)
        source = src.read(1, window=from_bounds(*src_bounds, src.transform))

        return source

        if source.size == 0:
            raise ValueError("Source data is empty after reading the window.")

        _, width, height = calculate_default_transform(
            src.crs, meta.get("crs"), src.width, src.height, *src_bounds
        )

        destination = np.empty(
            (1, meta.get("height"), meta.get("width")), dtype=meta.get("dtype")
        )

        destination, *_ = reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=meta.get("transform"),
            dst_crs=meta.get("crs"),
        )

        return destination
