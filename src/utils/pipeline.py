from typing import List, Tuple

from numpy import array, dot, histogram, minimum, ndarray, percentile, stack
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .dataset import ForestDataset
from .preprocessing import Preprocessor


def get_normalized_sampling_weights(
    bins: Series,
    probs: ndarray,
) -> ndarray:
    bins = stack(bins.apply(array))

    weights = dot(bins, probs)

    weights /= weights.sum()

    return weights


def get_splits(
    patches: DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    train_df, rest = train_test_split(
        patches, test_size=val_size + test_size, random_state=random_state
    )

    val_df, test_df = train_test_split(
        rest, test_size=test_size, random_state=random_state
    )

    return train_df, val_df, test_df


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
    sampling_weights: ndarray,
    batch_size: int,
    num_workers,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    sampler = WeightedRandomSampler(sampling_weights, len(sampling_weights))

    return (
        DataLoader(train_ds, batch_size, sampler=sampler, num_workers=num_workers),
        DataLoader(valid_ds, batch_size * 2, num_workers=num_workers),
        DataLoader(test_ds, batch_size * 2, num_workers=num_workers),
    )


def get_data(
    img_dir: str,
    patch_dir: str,
    gedi_dir: str,
    image_size: int,
    batch_size: int = 1,
    num_workers: int = 0,
    bins: List[int] = list(range(0, 55, 5)),
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Create preprocessor
    preprocessor = Preprocessor(img_dir, patch_dir, gedi_dir, image_size, bins)

    # Run preprocessor
    preprocessor.run()

    # Get labels
    labels = preprocessor.gedi.rh98

    # Get patches
    patches = preprocessor.patches

    # Get label distribution
    dist, _ = histogram(labels, bins)

    # Compute inverse probability weights
    probs = 1 / (dist + 1e-5)

    # Clip weights to avoid focussing on outliers
    max_weight = percentile(probs, 95)
    probs = minimum(probs, max_weight)

    # Create splits
    train_df, val_df, test_df = get_splits(patches)

    # Calculate weight for each patch of the training set
    weights = get_normalized_sampling_weights(train_df.bins, probs)

    # Create datasets
    train_ds, val_ds, test_ds = get_datasets(
        train_df, val_df, test_df, f"{patch_dir}/{image_size}"
    )

    # Create dataloaders
    train_dl, val_dl, test_dl = get_dataloaders(
        train_ds, val_ds, test_ds, weights, batch_size, num_workers
    )

    return train_dl, val_dl, test_dl
