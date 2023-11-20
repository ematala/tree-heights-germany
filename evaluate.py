import logging
import os
from argparse import ArgumentParser

from dotenv import load_dotenv
from pandas import DataFrame

from models import get_all_models, make_model
from utils.loss import loss
from utils.misc import get_device, seed_everyting
from utils.models import test
from utils.pipeline import get_data
from utils.plots import (
    plot_predictions,
    plot_true_vs_predicted_histogram,
    plot_true_vs_predicted_scatter,
)
from utils.predictions import predict_batch


def main():
    args = get_evaluation_args()
    image_size = 256
    random_state = 42
    num_workers = os.cpu_count() // 2
    bins = list(range(0, 55, 5))
    device = get_device()
    config = vars(args)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Set random seed
    seed_everyting(random_state)

    img_dir = config.get("img_dir")
    patch_dir = config.get("patch_dir")
    gedi_dir = config.get("gedi_dir")
    weights_dir = config.get("weights_dir")
    results_dir = config.get("results_dir")
    batch_size = config.get("batch_size")
    results_filename = config.get("filename")

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

    models = {
        name: make_model(name).load(os.path.join(weights_dir, f"{name}.pt")).to(device)
        for name in get_all_models()
        if os.path.exists(os.path.join(weights_dir, f"{name}.pt"))
    }

    # Initialize results DataFrame
    results = DataFrame(columns=["Test Loss", "MAE", "RMSE", "Test Loss By Range"])

    # Test each model
    for model_name, model in models.items():
        logging.info(f"Testing model {model_name}")
        metrics = test(model, test_dl, loss, device, bins)

        results.loc[model_name] = [
            metrics.get("total"),
            metrics.get("mae"),
            metrics.get("rmse"),
            metrics.get("loss_by_range"),
        ]

        logging.info(
            f"Test loss: {metrics.get('total'):>8f}\n"
            f"MAE: {metrics.get('mae'):>8f}\n"
            f"RMSE: {metrics.get('rmse'):>8f}\n"
            f"Ranges: {bins}\n"
            f"Losses by range: {metrics.get('loss_by_range')}"
        )

        plot_true_vs_predicted_histogram(
            metrics.get("targets"),
            metrics.get("predicted"),
            model_name,
            os.path.join(results_dir, f"{model_name}-histogram.pdf"),
        )

        plot_true_vs_predicted_scatter(
            metrics.get("targets"),
            metrics.get("predicted"),
            model_name,
            os.path.join(results_dir, f"{model_name}-scatter.pdf"),
        )

    plot_predictions(
        *predict_batch(models, test_dl, device),
        os.path.join(results_dir, "patches.pdf"),
    )

    # Save results
    results.to_csv(os.path.join(results_dir, results_filename), sep=";")


def get_evaluation_args():
    """Get arguments from command line

    Returns:
        Namespace: Arguments
    """

    load_dotenv()
    parser = ArgumentParser(
        description="Evaluate a model suite on predicting tree canopy heights"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=os.getenv("IMG_DIR", "data/images"),
        help="Path to images directory [default: data/images]",
    )
    parser.add_argument(
        "--patch_dir",
        type=str,
        default=os.getenv("PATCH_DIR", "data/patches"),
        help="Path to patches directory [default: data/patches]",
    )
    parser.add_argument(
        "--gedi_dir",
        type=str,
        default=os.getenv("GEDI_DIR", "data/gedi"),
        help="Path to GEDI directory [default: data/gedi]",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=os.getenv("WEIGHTS_DIR", "weights"),
        help="Path to weights directory [default: weights]",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.getenv("RESULTS_DIR", "results"),
        help="Path to results directory [default: results]",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size [default: 64]",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="evaluation.csv",
        help="Filename for the evaluation results [default: evaluation.csv]",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
