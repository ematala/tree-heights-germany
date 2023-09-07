import gc
import logging
import os
import re
from typing import List

from geopandas import GeoDataFrame, points_from_xy
from h5py import File as HDF5File
from numpy import any as npany
from numpy import float32
from numpy import sum as npsum
from pandas import DataFrame, MultiIndex, read_feather
from rasterio import open as ropen
from rasterio.features import rasterize
from shapely.geometry import box
from tqdm import tqdm

from src.utils.misc import get_bins, get_window_bounds


class Preprocessor:
    def __init__(
        self,
        patches_file: str = "data/patches/256/info.fth",
        img_dir: str = "data/images",
        patch_dir: str = "data/patches/256",
        gedi_file: str = "data/gedi/gedi_complete.fth",
        patch_size=256,
        image_size=4096,
        bins: List[int] = list(range(0, 55, 5)),
    ):
        self.patches_file = patches_file
        self.img_dir = img_dir
        self.patch_dir = patch_dir
        self.gedi_file = gedi_file
        self.patch_size = patch_size
        self.image_size = image_size
        self.n_patches = (self.image_size // self.patch_size) ** 2
        self.images = []
        self.gedi = None
        self.bins = bins
        self.patches = DataFrame(
            index=MultiIndex.from_product([[], []], names=["image", "patch"]),
            columns=[
                "labels",
                "bins",
            ],
            dtype="object",
        )

        logging.basicConfig(level=logging.INFO)

    def _validate_directories(self):
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory {self.img_dir} does not exist.")

        if not os.path.exists(self.patch_dir):
            raise FileNotFoundError(f"Patch directory {self.patch_dir} does not exist.")

        if not os.path.exists(self.gedi_file):
            raise FileNotFoundError(f"GEDI file {self.gedi_file} does not exist.")

    def _load_images(self):
        self.images = [
            f[: -len(".tif")]
            for f in os.listdir(self.img_dir)
            if re.match(r"L15\-\d{4}E\-\d{4}N\.tif", f)
        ]
        if len(self.images) == 0:
            raise ValueError("No images found in the image directory.")

    def _load_gedi(self):
        self.gedi = read_feather(
            self.gedi_file,
            columns=[
                "latitude",
                "longitude",
                "rh98",
                "solar_elevation",
                "quality_flag",
            ],
        )
        self.gedi = self.gedi[
            (self.gedi.solar_elevation < 0) & (self.gedi.quality_flag == 1)
        ]
        self.gedi = (
            GeoDataFrame(
                self.gedi,
                geometry=points_from_xy(self.gedi.longitude, self.gedi.latitude),
                crs="EPSG:32632",
            )
            .drop(
                columns=[
                    "latitude",
                    "longitude",
                    "solar_elevation",
                    "quality_flag",
                ]
            )
            .to_crs("EPSG:3857")
        )

    def _process_images(self):
        for image in tqdm(self.images):
            with ropen(os.path.join(self.img_dir, f"{image}.tif")) as src:
                bounds = box(*src.bounds)
                subset = self.gedi[self.gedi.geometry.intersects(bounds)]
                shapes = [(row.geometry, row.rh98) for row in subset.itertuples()]

                try:
                    mask = rasterize(
                        shapes=shapes,
                        out_shape=src.shape,
                        transform=src.transform,
                        fill=0,
                        all_touched=True,
                        dtype=float32,
                    )
                except ValueError as e:
                    logging.error(f"Error rasterizing image {image}: {str(e)}")
                    continue

                # Read the entire image into memory
                img = src.read([3, 2, 1, 4])

                subdir = os.path.join(self.patch_dir, image)
                os.makedirs(subdir)

                for patch in range(self.n_patches):
                    bounds = get_window_bounds(patch, self.patch_size)
                    row_start, row_end, col_start, col_end = bounds

                    data = img[:, row_start:row_end, col_start:col_end]
                    label = mask[row_start:row_end, col_start:col_end]

                    if npany(label):
                        # Save the image patch and label patch in a HDF5 file
                        with HDF5File(
                            f"{subdir}/{patch}.h5",
                            "w",
                        ) as hf:
                            hf.create_dataset("image", data=data)
                            hf.create_dataset("label", data=label)

                        self.patches.loc[(image, patch), :] = [
                            npsum(label != 0),
                            get_bins(label, self.bins),
                        ]

        gc.collect()

    def _save_patches(self):
        self.patches.reset_index().to_feather(self.patches_file)

    def run(self):
        logging.info("Starting preprocessing...")
        self._validate_directories()
        logging.info("Directories validated.")
        self._load_images()
        logging.info("Images loaded.")
        logging.info(f"Number of images: {len(self.images)}")
        if os.path.exists(self.patches_file):
            self.patches = read_feather(self.patches_file).set_index(["image", "patch"])
            logging.info("Loaded existing patch info file. Skipping image processing.")
            return
        self._load_gedi()
        logging.info("GEDI data loaded.")
        self._process_images()
        logging.info("Images processed.")
        logging.info(f"Number of patches: {self.patches.shape[0]}")
        logging.info(f"Number of labels: {self.patches.n_labels.sum()}")
        self._save_patches()
        logging.info("Patch info saved.")
        logging.info("Done.")


if __name__ == "__main__":
    Preprocessor().run()
