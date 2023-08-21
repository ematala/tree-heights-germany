import os
from datetime import timedelta
from time import time

from segmentation_models_pytorch import Unet
from sklearn.model_selection import train_test_split as split
from torch import device as Device
from torch.backends.mps import is_available as mps_available
from torch.cuda import is_available as cuda_available
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.utils.dataset import ForestDataset
from src.utils.loss import loss
from src.utils.models import evaluation, save, training
from src.utils.preprocessing import Preprocessor

img_dir = "data/images"
patch_dir = "data/patches/64"
log_dir = "logs"
model_dir = "models"
gedi_file = "data/gedi/gedi_complete.fth"
patches_file = "data/patches/64/info.fth"
seed = 42
batch_size = 16
patch_size = 64
num_workers = 6
learning_rate = 1e-4
epochs = 2
device = Device("cuda" if cuda_available() else "mps" if mps_available() else "cpu")

print(f"Using {device} device")

# Create preprocessor
preprocessor = Preprocessor(patches_file, img_dir, patch_dir, gedi_file, patch_size)

preprocessor.run()

# Extract patches
patches = preprocessor.patches.sample(frac=0.5, random_state=seed)

print(f"Total number of patches: {len(patches)}")

# Split patches
train, rest = split(patches, test_size=0.3, random_state=seed)
val, test = split(rest, test_size=0.5, random_state=seed)

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
    encoder_weights=None,
    in_channels=4,
).to(device)

print(model)

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

print(f"Training completed in {timedelta(seconds=(end - start))}")

score = evaluation(test_loader, model, loss, device)

print(f"Final loss on test set: {score}")

save(model, os.path.join(model_dir, f"{model.name}.pt"))
