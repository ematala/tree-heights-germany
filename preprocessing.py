import argparse
import logging
import os
import re
import uuid
from functools import partial
from itertools import chain
from multiprocessing import get_context
from typing import List

import numpy as np
import torch
from dotenv import load_dotenv
from geopandas import GeoDataFrame, points_from_xy
from h5py import File as HDF5File
from pandas import DataFrame, read_feather
from patchify import patchify
from rasterio import open as ropen
from rasterio.features import rasterize
from shapely.geometry import box
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import ForestDataset
from utils.misc import (
    get_label_bins,
    get_num_processes_to_spawn,
    send_telegram_message,
)


class Preprocessor:
    def __init__(
        self,
        img_dir: str = "data/images",
        patch_dir: str = "data/patches",
        gedi_dir: str = "data/gedi",
        patch_size: int = 256,
        patch_overlap: int = 128,
        bins: List[int] = list(range(0, 55, 5)),
        min_labels_per_patch: int = 5,
        **kwargs,
    ):
        self.img_dir = img_dir
        self.patch_dir = f"{patch_dir}/{patch_size}"
        self.patches_file = f"{self.patch_dir}/info.fth"
        self.gedi_file = f"{gedi_dir}/gedi_complete.fth"
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.bins = bins
        self.min_labels_per_patch = min_labels_per_patch
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
        patch_overlap: int,
        bins: List[int],
        min_labels_per_patch: int,
    ):
        try:
            valid_patches = []
            with ropen(os.path.join(img_dir, f"{image}.tif")) as src:
                bounds = box(*src.bounds)
                subset = gedi[gedi.geometry.intersects(bounds)]

                if subset.empty:
                    return []

                shapes = [(row.geometry, row.rh98) for row in subset.itertuples()]

                mask = rasterize(
                    shapes=shapes,
                    out_shape=src.shape,
                    transform=src.transform,
                    fill=0,
                    all_touched=True,
                    dtype=gedi.rh98.dtype,
                )

                # Read image from src and swap order of bands
                img = src.read([3, 2, 1, 4]).transpose((1, 2, 0))

                # Create patches
                patches = patchify(
                    img, (patch_size, patch_size, img.shape[-1]), patch_overlap
                ).squeeze()
                # Create patches for labels
                labels = patchify(
                    mask, (patch_size, patch_size), patch_overlap
                ).squeeze()

                rows, cols = patches.shape[:2]

                subdir = os.path.join(patch_dir, image)
                os.makedirs(subdir, exist_ok=True)

                for row in range(rows):
                    for col in range(cols):
                        data = patches[row, col].transpose((2, 0, 1))
                        label = labels[row, col]

                        patch = str(uuid.uuid4())

                        if np.count_nonzero(label) >= min_labels_per_patch:
                            with HDF5File(f"{subdir}/{patch}.h5", "w") as hf:
                                hf.create_dataset("image", data=data)
                                hf.create_dataset("label", data=label)

                            valid_patches.append(
                                {
                                    "image": image,
                                    "patch": patch,
                                    "labels": np.sum(label != 0),
                                    "bins": get_label_bins(label, bins),
                                }
                            )

            return valid_patches
        except Exception as e:
            logging.info(f"Error processing image {image}: {e}")
            return []

    def _process_images(self):
        fn = partial(
            self._process_single_image,
            gedi=self.gedi,
            img_dir=self.img_dir,
            patch_dir=self.patch_dir,
            patch_size=self.patch_size,
            bins=self.bins,
            patch_overlap=self.patch_overlap,
            min_labels_per_patch=self.min_labels_per_patch,
        )

        results = []

        num_processes = get_num_processes_to_spawn(len(self.images))

        with get_context("spawn").Pool(num_processes) as pool:
            for result in tqdm(
                pool.imap_unordered(fn, self.images),
                "Processing images",
                len(self.images),
            ):
                results.append(result)
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


def get_preprocessing_args():
    """Get arguments from command line

    Returns:
        Namespace: Arguments
    """

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Preprocess images and GEDI data for training and evaluation."
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Patch size for the images [default: 256]",
    )

    parser.add_argument(
        "--patch_overlap",
        type=int,
        default=128,
        help="Overlap between patches [default: 128]",
    )

    parser.add_argument(
        "--img_dir",
        type=str,
        default=os.getenv("IMG_DIR", "data/images"),
        help="Directory containing the images [default: data/images]",
    )

    parser.add_argument(
        "--patch_dir",
        type=str,
        default=os.getenv("PATCH_DIR", "data/patches"),
        help="Directory to store the patches [default: data/patches]",
    )

    parser.add_argument(
        "--gedi_dir",
        type=str,
        default=os.getenv("GEDI_DIR", "data/gedi"),
        help="Directory containing the GEDI data [default: data/gedi]",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for calculating dataset stats [default: 128]",
    )

    return parser.parse_args()


def get_dataset_stats(
    patches: DataFrame,
    patch_dir: str,
    patch_size: int,
    batch_size: int,
    num_workers: int = 4,
    **kwargs,
):
    dataset = ForestDataset(
        patches=patches,
        patch_dir=f"{patch_dir}/{patch_size}",
        apply_transforms=False,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    channel_sum, channel_squared_sum, num_batches = 0, 0, 0

    with torch.no_grad():
        for img, _ in tqdm(loader, "Calculating dataset stats"):
            channel_sum += torch.mean(img, dim=[0, 2, 3])
            channel_squared_sum += torch.mean(img**2, dim=[0, 2, 3])
            num_batches += 1

    channel_mean = channel_sum / num_batches
    chanel_std = (channel_squared_sum / num_batches - channel_mean**2) ** 0.5

    stats = (
        f"patches: {len(patches)}\n"
        f"labels: {patches.labels.sum()}\n"
        f"channel mean: {channel_mean}\n"
        f"channel std: {chanel_std}\n"
    )

    logging.info(stats)

    return stats


def main():
    args = get_preprocessing_args()
    config = vars(args)
    preprocessor = Preprocessor(**config)
    preprocessor.run()
    stats = get_dataset_stats(preprocessor.patches, **config)
    send_telegram_message(stats)


if __name__ == "__main__":
    main()
