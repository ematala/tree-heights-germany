import os

from numpy import argmax
from sklearn.model_selection import train_test_split as split
from torch import device as Device
from torch.backends.mps import is_available as mps_available
from torch.cuda import is_available as cuda_available
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from ...utils import (
    ForestDataset,
    Preprocessor,
    compute_sampling_weights,
    loss,
    save,
    seed_everyting,
    test,
    train,
)
from . import VitNet

if __name__ == "__main__":
    patch_size = 256
    img_dir = os.getenv("IMG_DIR")
    model_dir = os.getenv("MODEL_DIR")
    patch_dir = os.getenv("PATCH_DIR")
    gedi_file = os.getenv("GEDI_DIR")
    random_state = 42
    batch_size = 12
    num_workers = os.cpu_count()
    learning_rate = 1e-4
    epochs = 25
    bins = list(range(0, 55, 5))
    device = Device("cuda" if cuda_available() else "mps" if mps_available() else "cpu")

    seed_everyting(random_state)

    print("Starting training")

    print(f"Using {device} device")

    # Create preprocessor
    preprocessor = Preprocessor(img_dir, patch_dir, gedi_file, patch_size)

    # Run preprocessor
    preprocessor.run()

    # Get labels
    labels = preprocessor.gedi.rh98

    # Get patches
    patches = preprocessor.patches

    print(f"Total number of patches: {len(patches)}")

    # Create stratification
    stratify = patches.bins.apply(argmax)

    # Split patches
    train_patches, rest_patches = split(
        patches, test_size=0.3, random_state=random_state, stratify=stratify
    )

    # Create stratification for rest
    stratify = stratify[patches.index.isin(rest_patches.index)]

    # Split rest
    val_patches, test_patches = split(
        rest_patches, test_size=0.5, random_state=random_state, stratify=stratify
    )

    # Create datasets
    train_data = ForestDataset(train_patches, f"{patch_dir}/{patch_size}")
    val_data = ForestDataset(val_patches, f"{patch_dir}/{patch_size}")
    test_data = ForestDataset(test_patches, f"{patch_dir}/{patch_size}")

    # Create weighted sampler
    weights = compute_sampling_weights(train_patches, labels, bins)
    sampler = WeightedRandomSampler(weights, len(train_patches))

    # Create dataloaders
    train_loader = DataLoader(
        train_data, batch_size, False, sampler, num_workers=num_workers
    )
    val_loader = DataLoader(val_data, batch_size, False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size, False, num_workers=num_workers)

    model = VitNet(hidden_size=256).to(device)

    # Create optimizer
    optimizer = AdamW(model.parameters(), learning_rate)

    # Create scheduler
    scheduler = CosineAnnealingLR(optimizer, epochs)

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train(train_loader, model, loss, device, optimizer, scheduler)
        test(val_loader, model, loss, device)

    # Save model
    print(f"Saving model {model.name}")

    model_file = os.path.join(model_dir, f"m-{model.name}-p{patch_size}-e{epochs}.pt")

    save(model, model_file)
