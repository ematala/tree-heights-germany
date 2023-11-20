import os
from typing import Tuple

from h5py import File as HDF5File
from numpy import ndarray
from pandas import DataFrame
from torch import Tensor
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
    A Pytorch Dataset to load and preprocess satellite images and corresponding
    labels for a tree height regression task.
    """

    def __init__(
        self,
        patches: DataFrame,
        patch_dir: str = "data/patches/256",
    ):
        """
        Args:

        patches (DataFrame): A DataFrame containing the patches to be loaded.
        patch_dir (str): Directory with all the patches.
        """

        self.patch_dir = patch_dir
        self.patches = patches

    def transform(self, img: ndarray, label: ndarray) -> Tuple[Tensor, Tensor]:
        img = add_ndvi(img)
        img, label = to_tensor(img), to_tensor(label)
        img = normalize(img)
        img, label = random_vertical_flip(img, label, prob=0.5)
        img, label = random_horizontal_flip(img, label, prob=0.5)
        img, label = random_rotation(img, label, prob=0.5)

        return img, label

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
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

        return self.transform(img, label)
