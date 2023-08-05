import os
from typing import Tuple

from h5py import File as HDF5File
from numpy import float32
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
        patches: DataFrame,
        patch_dir: str = "data/patches",
    ):
        """
        Args:

        patches (DataFrame): A DataFrame containing the patches to be loaded.
        patch_dir (str): Directory with all the patches.
        """

        self.patch_dir = patch_dir
        self.patches = patches

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
        with HDF5File(os.path.join(self.patch_dir, f"{image}/{patch}.h5")) as hf:
            img = hf["image"][:]
            label = hf["label"][:]

        # Normalize the image patch
        img = (img - img.min()) / (img.max() - img.min())

        # Return the image patch and label patch as PyTorch tensors
        return (
            from_numpy(img.astype(float32)).float(),
            from_numpy(label.astype(float32)).float(),
        )
