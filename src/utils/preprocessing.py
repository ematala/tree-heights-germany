import logging
import os
import re
from argparse import ArgumentParser
from itertools import chain
from multiprocessing import get_context
from typing import List

from geopandas import GeoDataFrame, points_from_xy
from h5py import File as HDF5File
from numpy import any as npany
from numpy import float32
from numpy import sum as npsum
from pandas import DataFrame, read_feather
from rasterio import open as ropen
from rasterio.features import rasterize
from shapely.geometry import box
from tqdm import tqdm

from .misc import get_label_bins, get_normalized_image, get_window_bounds


class Preprocessor:
    def __init__(
        self,
        img_dir: str = "data/images",
        patch_dir: str = "data/patches",
        gedi_dir: str = "data/gedi",
        patch_size: int = 256,
        image_size: int = 4096,
        bins: List[int] = list(range(0, 55, 5)),
    ):
        self.img_dir = img_dir
        self.patch_dir = f"{patch_dir}/{patch_size}"
        self.patches_file = f"{self.patch_dir}/info.fth"
        self.gedi_file = f"{gedi_dir}/gedi_complete.fth"
        self.patch_size = patch_size
        self.image_size = image_size
        self.bins = bins
        self.n_patches = (self.image_size // self.patch_size) ** 2
        self.images = []
        self.gedi = None
        self.patches = None

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    def _validate_directories(self):
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory {self.img_dir} does not exist.")

        if not os.path.exists(self.patch_dir):
            logging.warning(
                f"Patch directory {self.patch_dir} did not exist. Creating it now."
            )
            os.makedirs(self.patch_dir)

        if not os.path.exists(self.gedi_file):
            raise FileNotFoundError(f"GEDI file {self.gedi_file} does not exist.")

    def _load_images(self):
        regexp = r"L15\-\d{4}E\-\d{4}N\.tif"

        self.images = [
            f[: -len(".tif")] for f in os.listdir(self.img_dir) if re.match(regexp, f)
        ]

        if len(self.images) == 0:
            raise ValueError("No images found in the image directory.")

    def _load_gedi(self):
        self.gedi = read_feather(self.gedi_file)

        self.gedi = self.gedi[
            (self.gedi.solar_elevation < 0)
            & (self.gedi.quality_flag == 1)
            & (self.gedi.num_detectedmodes > 0)
            & (self.gedi.degrade_flag == 0)
            & (self.gedi.stale_return_flag == 0)
            & (self.gedi.landsat_treecover > 0)
            & (self.gedi.rh98 >= 0)
            & (self.gedi.rh98 <= 70)
        ]

        self.gedi = GeoDataFrame(
            self.gedi.rh98,
            geometry=points_from_xy(self.gedi.longitude, self.gedi.latitude),
            crs="EPSG:32632",
        ).to_crs("EPSG:3857")

    def _process_single_image(
        self,
        image: str,
        gedi: GeoDataFrame,
        img_dir: str,
        patch_dir: str,
        patch_size: int,
        bins: List[int],
        n_patches: int,
    ):
        valid_patches = []
        with ropen(os.path.join(img_dir, f"{image}.tif")) as src:
            bounds = box(*src.bounds)
            subset = gedi[gedi.geometry.intersects(bounds)]

            if subset.empty:
                logging.warning(f"No valid geometries found for image {image}.")
                return []

            shapes = [(row.geometry, row.rh98) for row in subset.itertuples()]

            mask = rasterize(
                shapes=shapes,
                out_shape=src.shape,
                transform=src.transform,
                fill=0,
                all_touched=True,
                dtype=float32,
            )

            # Read image from src
            img = get_normalized_image(src)

            subdir = os.path.join(patch_dir, image)
            os.makedirs(subdir, exist_ok=True)

            for patch in range(n_patches):
                bounds = get_window_bounds(patch, patch_size)
                row_start, row_end, col_start, col_end = bounds

                data = img[:, row_start:row_end, col_start:col_end]
                labels = mask[row_start:row_end, col_start:col_end]

                if npany(labels):
                    with HDF5File(f"{subdir}/{patch}.h5", "w") as hf:
                        hf.create_dataset("image", data=data)
                        hf.create_dataset("labels", data=labels)

                    valid_patches.append(
                        {
                            "image": image,
                            "patch": patch,
                            "labels": npsum(labels != 0),
                            "bins": get_label_bins(labels, bins),
                        }
                    )
        return valid_patches

    def _process_images(self):
        img_args = [
            (
                image,
                self.gedi,
                self.img_dir,
                self.patch_dir,
                self.patch_size,
                self.bins,
                self.n_patches,
            )
            for image in self.images
        ]
        with get_context("spawn").Pool(os.cpu_count() // 2) as pool:
            results = pool.starmap(
                self._process_single_image,
                tqdm(img_args, "Processing images", len(img_args)),
            )

        flat_results = list(chain.from_iterable(results))
        self.patches = DataFrame(flat_results).set_index(["image", "patch"])

    def _save_patches(self):
        self.patches.reset_index().to_feather(self.patches_file)

    def run(self):
        logging.info("Starting preprocessing...")
        self._validate_directories()
        logging.info("Directories validated.")
        self._load_images()
        logging.info("Images loaded.")
        logging.info(f"Number of images: {len(self.images)}")
        self._load_gedi()
        logging.info("GEDI data loaded.")
        if os.path.exists(self.patches_file):
            self.patches = read_feather(self.patches_file).set_index(["image", "patch"])
            logging.info("Loaded existing patch info file. Skipping image processing.")
            logging.info(f"Number of patches: {len(self.patches)}")
            logging.info(f"Number of labels: {self.patches.labels.sum()}")
            return self
        logging.info("Starting image processing.")
        self._process_images()
        logging.info("Images processed.")
        logging.info(f"Number of patches: {len(self.patches)}")
        logging.info(f"Number of labels: {self.patches.labels.sum()}")
        self._save_patches()
        logging.info("Patch info saved.")
        logging.info("Done.")

        return self


def get_args():
    """Get arguments from command line

    Returns:
        Namespace: Arguments
    """
    parser = ArgumentParser(
        description="Preprocess images and GEDI data for training and evaluation."
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Patch size for the images [default: 256]",
    )
    return parser.parse_args()


def main():
    args = get_args()
    Preprocessor(
        img_dir=os.getenv("IMG_DIR"),
        patch_dir=os.getenv("PATCH_DIR"),
        gedi_dir=os.getenv("GEDI_DIR"),
        **vars(args),
    ).run()


if __name__ == "__main__":
    main()
