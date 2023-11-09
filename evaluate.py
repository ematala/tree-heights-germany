import logging
import os
from argparse import ArgumentParser

from dotenv import load_dotenv
from pandas import DataFrame

from utils import (
    get_data,
    get_device,
    load_model,
    loss,
    plot_predictions,
    plot_true_vs_predicted_histogram,
    plot_true_vs_predicted_scatter,
    predict_batch,
    seed_everyting,
    test,
)


def get_args():
    """Get arguments from command line

    Returns:
        Namespace: Arguments
    """
    parser = ArgumentParser(
        description="Evaluate a model suite on predicting tree canopy heights"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size [default: 16]",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="evaluation.csv",
        help="Filename for the evaluation results [default: evaluation.csv]",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    img_dir = os.getenv("IMG_DIR")
    model_dir = os.getenv("MODEL_DIR")
    patch_dir = os.getenv("PATCH_DIR")
    results_dir = os.getenv("RESULTS_DIR")
    gedi_dir = os.getenv("GEDI_DIR")
    image_size = 256
    random_state = 42
    num_workers = os.cpu_count() // 2
    bins = list(range(0, 55, 5))
    device = get_device()
    config = get_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Set random seed
    seed_everyting(random_state)

    batch_size = config.batch_size
    results_filename = config.filename

    # Get data
    _, _, test_dl = get_data(
        img_dir,
        patch_dir,
        gedi_dir,
        image_size,
        batch_size,
        num_workers,
        bins,
    )

    # Load all models
    models = {
        filename[: -len(".pt")]: load_model(os.path.join(model_dir, filename), device)
        for filename in [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    }

    # Initialize results DataFrame
    results = DataFrame(columns=["Test Loss", "MAE", "RMSE", "Test Loss By Range"])

    # Test each model
    for model_name, model in models.items():
        logging.info(f"Testing model {model_name}")
        (
            test_loss,
            test_mae,
            test_rmse,
            test_loss_by_range,
            labels,
            predictions,
        ) = test(model, test_dl, loss, device, bins)

        results.loc[model_name] = [test_loss, test_mae, test_rmse, test_loss_by_range]

        logging.info(
            f"Test loss: {test_loss:>8f}\n"
            f"MAE: {test_mae:>8f}\n"
            f"RMSE: {test_rmse:>8f}\n"
            f"Ranges: {bins}\n"
            f"Losses by range: {test_loss_by_range}"
        )

        plot_true_vs_predicted_histogram(
            labels,
            predictions,
            model_name,
            os.path.join(results_dir, f"{model_name}-histogram.pdf"),
        )

        plot_true_vs_predicted_scatter(
            labels,
            predictions,
            model_name,
            os.path.join(results_dir, f"{model_name}-scatter.pdf"),
        )

    plot_predictions(
        *predict_batch(models, test_dl, device),
        os.path.join(results_dir, "patches.pdf"),
    )

    # Save results
    results.to_csv(os.path.join(results_dir, results_filename), sep=";")


if __name__ == "__main__":
    main()
