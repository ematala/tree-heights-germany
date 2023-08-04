import os
from typing import Tuple

import numpy as np
from h5py import File as HDF5File
from pandas import DataFrame
from torch import Tensor, from_numpy
from torch.utils.data import Dataset


class ForestDataset(Dataset):

    """
    A Pytorch Dataset to load and preprocess satellite images and corresponding
    labels for a tree height regression task.
    """

    def __init__(
        self,
        img_dir: str,
        labels_dir: str,
        patches: DataFrame,
        patch_size: int = 64,
        image_size: int = 4096,
    ):
        """
        Args:

        img_dir (str): Directory with all the images.
        labels_dir (str): Directory with all the labels.
        patch_size (int): The size of the patches to be extracted from the images.
        image_size (int): The size of the images.
        patches (DataFrame): A DataFrame containing the patches to be loaded.
        """

        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.patch_size = patch_size
        self.image_size = image_size
        self.patches = patches

    def getrowcol(self, idx: int) -> Tuple[int, int]:
        """
        This function returns the row and column of a patch given its id.
        Args:

        idx (int): The id of the patch.
        """
        row = (idx // (self.image_size // self.patch_size)) * self.patch_size
        col = (idx % (self.image_size // self.patch_size)) * self.patch_size

        return row, col

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

        # Open the HDF5 file for the patch
        with HDF5File(os.path.join(self.labels_dir, f"{image}/{patch}.h5")) as hf:
            img = hf["image"][:]
            label = hf["label"][:]

        # Normalize the image patch
        # img = (img - img.min()) / (img.max() - img.min())
        # img = img.transpose((1, 2, 0))

        # Return the image patch and label patch as PyTorch tensors
        return (
            from_numpy(img.astype(np.float32)).float(),
            from_numpy(label.astype(np.float32)).float(),
        )
