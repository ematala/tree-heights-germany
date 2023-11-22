import os
from typing import Tuple

import torch
from h5py import File as HDF5File
from numpy import ndarray
from pandas import DataFrame
from torch.utils.data import Dataset

from .transforms import (
    add_ndvi,
    normalize,
    random_horizontal_flip,
    random_rotation,
    random_vertical_flip,
    to_tensor,
)


class ForestDataset(Dataset):

    """
    A Pytorch Dataset to load and transform multispectral satellite images and
    labels for predicting tree canopy height.
    """

    def __init__(
        self,
        patches: DataFrame,
        patch_dir: str,
        apply_transforms: bool = True,
        apply_augmentations: bool = False,
        **kwargs,
    ):
        """
        Args:

        patches (DataFrame): A DataFrame containing the patches to be loaded.
        patch_dir (str): Directory with all the patches.
        """

        self.patch_dir = patch_dir
        self.patches = patches
        self.apply_transforms = apply_transforms
        self.apply_augmentations = apply_augmentations

    def transform(
        self, img: ndarray, label: ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img, label = add_ndvi(img), label
        img, label = to_tensor(img), to_tensor(label)
        img, label = normalize(img), label

        return img, label

    def augment(self, img: ndarray, label: ndarray):
        img, label = random_vertical_flip(img, label, prob=0.5)
        img, label = random_horizontal_flip(img, label, prob=0.5)
        img, label = random_rotation(img, label, prob=0.5)

        return img, label

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This function loads an image, preprocesses it and returns it along
        with the corresponding label

        Args:

        idx (int): The index of the image to load.
        """

        image, patch = self.patches.index[idx]

        filename = os.path.join(self.patch_dir, image, f"{patch}.h5")

        # Open the HDF5 file for the patch
        with HDF5File(filename) as hf:
            img = hf["image"][:]
            label = hf["label"][:]

        if self.apply_transforms and self.apply_augmentations:
            return self.augment(*self.transform(img, label))

        if self.apply_transforms:
            return self.transform(img, label)

        if self.apply_augmentations:
            return self.augment(img, label)

        return to_tensor(img), to_tensor(label)
