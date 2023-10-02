import os

from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from ...utils import (
    get_data,
    get_device,
    loss,
    save,
    seed_everyting,
    test,
    train,
)
from . import Unet

if __name__ == "__main__":
    patch_size = 256
    img_dir = os.getenv("IMG_DIR")
    model_dir = os.getenv("MODEL_DIR")
    patch_dir = os.getenv("PATCH_DIR")
    gedi_file = os.getenv("GEDI_DIR")
    random_state = 42
    batch_size = 12
    num_workers = os.cpu_count()
    learning_rate = 1e-2
    epochs = 25
    bins = list(range(0, 55, 5))
    device = get_device()

    seed_everyting(random_state)

    print("Starting training")

    # Get data
    train_dl, val_dl, test_dl = get_data(
        img_dir,
        patch_dir,
        gedi_file,
        patch_size,
        batch_size,
        num_workers,
        bins,
    )

    model = Unet().to(device)

    # Create optimizer
    optimizer = SGD(model.parameters(), learning_rate)

    # Create scheduler
    scheduler = CosineAnnealingLR(optimizer, epochs)

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train(train_dl, model, loss, device, optimizer, scheduler)
        test(val_dl, model, loss, device)

    print("Training finished.")

    test(test_dl, model, loss, device)

    # Save model
    print(f"Saving model {model.name}")

    save(model, os.path.join(model_dir, f"{model.name}-{patch_size}.pt"))
