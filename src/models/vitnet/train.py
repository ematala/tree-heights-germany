import os

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ...utils import (
    Preprocessor,
    get_dataloaders,
    get_datasets,
    get_device,
    get_splits,
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
    gedi_file = os.getenv("GEDI_DIR")
    random_state = 42
    batch_size = 12
    num_workers = os.cpu_count()
    learning_rate = 1e-4
    epochs = 25
    bins = list(range(0, 55, 5))
    device = get_device()

    seed_everyting(random_state)

    print("Starting training")

    # Create preprocessor
    preprocessor = Preprocessor(img_dir, patch_dir, gedi_file, patch_size)

    # Run preprocessor
    preprocessor.run()

    # Get labels
    labels = preprocessor.gedi.rh98

    # Get patches
    patches = preprocessor.patches

    # Create splits
    train_df, val_df, test_df = get_splits(patches)

    # Create datasets
    train_ds, val_ds, test_ds = get_datasets(
        train_df, val_df, test_df, f"{patch_dir}/{patch_size}"
    )

    # Create dataloaders
    train_dl, val_dl, test_dl = get_dataloaders(
        train_ds, val_ds, test_ds, batch_size, num_workers
    )

    model = VitNet(
        image_size=patch_size,
        hidden_size=patch_size * 2,
        intermediate_size=patch_size * 4,
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

    # Save model
    print(f"Training finished. Saving model {model.name}")

    save(model, os.path.join(model_dir, f"{model.name}.pt"))
