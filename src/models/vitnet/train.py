import os

from torch.optim import AdamW
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
from . import VitNet

if __name__ == "__main__":
    patch_size = 64
    img_dir = os.getenv("IMG_DIR")
    model_dir = os.getenv("MODEL_DIR")
    patch_dir = os.getenv("PATCH_DIR")
    results_dir = os.getenv("RESULTS_DIR")
    gedi_file = os.getenv("GEDI_DIR")
    random_state = 42
    batch_size = 512
    num_workers = os.cpu_count()
    learning_rate = 1e-4
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

    # Create model
    model = VitNet(
        image_size=patch_size,
        hidden_size=patch_size * 4,
        intermediate_size=patch_size * 8,
    ).to(device)

    # Create optimizer
    optimizer = AdamW(model.parameters(), learning_rate)

    # Create scheduler
    scheduler = CosineAnnealingLR(optimizer, epochs)

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train(train_dl, model, loss, device, optimizer, scheduler)
        test(val_dl, model, loss, device)

    print("Training finished.")

    # Test model
    test(test_dl, model, loss, device)

    print(f"Saving model {model.name}")

    # Save model
    save(model, os.path.join(model_dir, f"{model.name}-{patch_size}-e{epochs}.pt"))
