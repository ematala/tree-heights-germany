import os

from sklearn.model_selection import train_test_split as split
from torch import device as Device
from torch import save
from torch.backends.mps import is_available as mps_available
from torch.cuda import is_available as cuda_available
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.unet import UNet
from src.utils.dataset import ForestDataset
from src.utils.loss import loss
from src.utils.models import train_loop, validation_loop
from src.utils.preprocessing import Preprocessor

img_dir = "data/images"
patch_dir = "data/patches"
log_dir = "logs"
model_dir = "models"
gedi_file = "data/gedi/gedi_complete.fth"
patches_file = "data/info/patches.fth"
seed = 42
batch_size = 64
num_workers = 6
learning_rate = 1e-4
epochs = 2
device = Device("cuda" if cuda_available() else "mps" if mps_available() else "cpu")
print(f"Using {device} device")


preprocessor = Preprocessor(patches_file, img_dir, patch_dir, gedi_file)

preprocessor.run()

patches = preprocessor.patches

print(f"Total number of patches: {len(patches)}")

train, rest = split(patches, test_size=0.3, random_state=seed)
val, test = split(rest, test_size=0.5, random_state=seed)

train_data = ForestDataset(train)
val_data = ForestDataset(val)
test_data = ForestDataset(test)

train_loader = DataLoader(train_data, batch_size, True, num_workers=num_workers)
val_loader = DataLoader(val_data, batch_size, False, num_workers=num_workers)
test_loader = DataLoader(val_data, batch_size, False, num_workers=num_workers)

model = UNet().to(device)

optimizer = Adam(model.parameters(), learning_rate)

writer = SummaryWriter(log_dir)

for epoch in range(epochs):
    train_loop(train_loader, model, loss, writer, device, epoch, optimizer)
    validation_loop(val_loader, model, loss, writer, device, epoch)

writer.close()

print("Training complete.")

save(model.state_dict(), os.path.join(model_dir, "unet.pt"))
