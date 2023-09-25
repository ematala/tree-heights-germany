# Imports
import os
from datetime import timedelta
from logging import INFO, basicConfig, info
from time import time

from numpy import argmax
from sklearn.model_selection import train_test_split as split
from torch import device as Device
from torch.backends.mps import is_available as mps_available
from torch.cuda import is_available as cuda_available
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from src.models.vitnet import VitNet
from src.utils.dataset import ForestDataset
from src.utils.loss import loss
from src.utils.misc import seed
from src.utils.models import evaluation, save, training
from src.utils.preprocessing import Preprocessor
from src.utils.sampling import compute_sampling_weights

patch_size = 256
img_dir = "data/images"
log_dir = "logs"
model_dir = "models"
patch_dir = f"data/patches/{patch_size}"
patches_file = f"data/patches/{patch_size}/info.fth"
gedi_file = "data/gedi/gedi_complete.fth"
random_state = 42
batch_size = 12
num_workers = 6
learning_rate = 1e-2
epochs = 5
bins = list(range(0, 55, 5))
device = Device("cuda" if cuda_available() else "mps" if mps_available() else "cpu")

seed(random_state)

basicConfig(level=INFO)

info(f"Using {device} device")

if __name__ == "__main__":
    info("Starting training")
    # Create preprocessor
    preprocessor = Preprocessor(patches_file, img_dir, patch_dir, gedi_file, patch_size)

    # Run preprocessor
    preprocessor.run()

    # Get labels
    labels = preprocessor.gedi.rh98

    # Get patches
    patches = preprocessor.patches

    info(f"Total number of patches: {len(patches)}")

    # Create stratification
    stratify = patches.bins.apply(argmax)

    # Split patches
    train, rest = split(
        patches, test_size=0.3, random_state=random_state, stratify=stratify
    )

    # Create stratification for rest
    stratify = stratify[patches.index.isin(rest.index)]

    # Split rest
    val, test = split(rest, test_size=0.5, random_state=random_state, stratify=stratify)

    # Create datasets
    traindata = ForestDataset(train, patch_dir)
    valdata = ForestDataset(val, patch_dir)
    testdata = ForestDataset(test, patch_dir)

    # Create weighted sampler
    weights = compute_sampling_weights(train, labels, bins)
    sampler = WeightedRandomSampler(weights, len(train))

    # Create dataloaders
    trainloader = DataLoader(
        traindata, batch_size, False, sampler, num_workers=num_workers
    )
    valloader = DataLoader(valdata, batch_size, False, num_workers=num_workers)
    testloader = DataLoader(testdata, batch_size, False, num_workers=num_workers)

    # Initialise model
    model = VitNet().to(device)

    model_file = os.path.join(model_dir, f"m-{model.name}-p{patch_size}-e{epochs}.pt")

    info(model)

    # Create optimizer
    optimizer = Adam(model.parameters(), learning_rate)

    # Create scheduler
    scheduler = CosineAnnealingLR(optimizer, epochs)

    # Create writer
    writer = SummaryWriter(log_dir)

    start = time()

    # Training loop
    for epoch in range(epochs):
        training(trainloader, model, loss, device, writer, epoch, optimizer)
        evaluation(valloader, model, loss, device, writer, epoch)
        scheduler.step()

    end = time()

    writer.close()

    info(f"Training completed in {timedelta(seconds=(end - start))}")

    score = evaluation(testloader, model, loss, device)

    info(f"Final loss on test set: {score}")

    info(f"Saving model {model.name}")

    # Save model
    save(model, model_file)
