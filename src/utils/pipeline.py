from typing import Tuple

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .dataset import ForestDataset


def get_splits(
    patches: DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    train, rest = train_test_split(
        patches, test_size=val_size + test_size, random_state=random_state
    )

    val, test = train_test_split(rest, test_size=test_size, random_state=random_state)

    return train, val, test


def get_datasets(
    train_df: DataFrame,
    val_df: DataFrame,
    test_df: DataFrame,
    patch_dir: str,
) -> Tuple[Dataset, Dataset, Dataset]:
    return (
        ForestDataset(train_df, patch_dir),
        ForestDataset(val_df, patch_dir),
        ForestDataset(test_df, patch_dir),
    )


def get_dataloaders(
    train_ds: Dataset,
    valid_ds: Dataset,
    test_ds: Dataset,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    return (
        DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(valid_ds, batch_size * 2, num_workers=num_workers),
        DataLoader(test_ds, batch_size * 2, num_workers=num_workers),
    )
