import os
from datetime import timedelta
from logging import INFO, basicConfig, info
from time import time

from numpy import digitize, percentile
from numpy.random import seed as npseed
from segmentation_models_pytorch import Unet
from sklearn.model_selection import train_test_split as split
from torch import device as Device
from torch.backends.mps import is_available as mps_available
from torch.cuda import is_available as cuda_available
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from src.models.unet import Unet
from src.utils.dataset import ForestDataset
from src.utils.loss import loss
from src.utils.models import evaluation, save, training
from src.utils.preprocessing import Preprocessor

patch_size = 128
img_dir = "data/images"
log_dir = "logs"
model_dir = "models"
patch_dir = f"data/patches/{patch_size}"
patches_file = f"data/patches/{patch_size}/info.fth"
gedi_file = "data/gedi/gedi_complete.fth"
seed = 42
batch_size = 16
num_workers = 6
learning_rate = 1e-4
epochs = 5
ranges = [(i, i + 5) for i in range(0, 50, 5)]
device = Device("cuda" if cuda_available() else "mps" if mps_available() else "cpu")

npseed(seed)
basicConfig(level=INFO, filename=os.path.join(log_dir, "main.log"), filemode="w")

info(f"Using {device} device")

if __name__ == "__main__":
    info("Starting training")
    # Create preprocessor
    preprocessor = Preprocessor(patches_file, img_dir, patch_dir, gedi_file, patch_size)

    preprocessor.run()

    # Extract patches
    patches = preprocessor.patches.sample(frac=0.5, random_state=seed)

    info(f"Total number of patches: {len(patches)}")

    # Create quantiles and stratify
    quantiles = percentile(patches.n_labels, [25, 50, 75])
    stratify = digitize(patches.n_labels, quantiles)

    # Split patches
    train, rest = split(patches, test_size=0.3, random_state=seed, stratify=stratify)

    stratify = stratify[patches.index.isin(rest.index)]

    val, test = split(rest, test_size=0.5, random_state=seed, stratify=stratify)

    # Create datasets
    train_data = ForestDataset(train, patch_dir)
    val_data = ForestDataset(val, patch_dir)
    test_data = ForestDataset(test, patch_dir)

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size, True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size, False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size, False, num_workers=num_workers)

    # Create model
    model = Unet(
        encoder_name="inceptionv4",
        encoder_weights=None,
        in_channels=4,
    ).to(device)

    info(model)

    # Create optimizer
    optimizer = Adam(model.parameters(), learning_rate)

    # Create writer
    writer = SummaryWriter(log_dir)

    # Training loop
    start = time()

    for epoch in range(epochs):
        training(train_loader, model, loss, device, writer, epoch, optimizer)
        evaluation(val_loader, model, loss, device, writer, epoch)

    end = time()

    writer.close()

    info(f"Training completed in {timedelta(seconds=(end - start))}")

    score = evaluation(test_loader, model, loss, device)

    info(f"Final loss on test set: {score}")

    info(f"Saving model {model.name}")

    save(model, os.path.join(model_dir, f"{model.name}.pt"))
