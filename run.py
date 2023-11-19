import argparse
import os
import re
from functools import partial

from torch.multiprocessing import get_context
from tqdm import tqdm

from models import get_all_models, make_model
from utils.io import save_prediction
from utils.misc import get_device
from utils.predictions import predict_image


def get_args():
    parser = argparse.ArgumentParser(description="Predict a single image")

    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Patch size for the images [default: 256]",
    )

    parser.add_argument(
        "--model",
        choices=get_all_models(),
        default="vit-tiny",
        help="Model to use for prediction [default: vit-tiny]",
    )

    parser.add_argument(
        "--weights_dir",
        type=str,
        default="weights",
        help="Path to the weights folder",
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="input",
        help="Path to the input folder",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Path to the output folder",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=5,
        help="Threshold for the predictions [default: 5]",
    )

    return parser.parse_args()


def predict_and_save_image(
    img: str,
    model: str,
    weights_dir: str,
    input_dir: str,
    output_dir: str,
    threshold: float = 5,
    patch_size: int = 256,
):
    device = get_device()

    model = make_model(model).load(os.path.join(weights_dir, f"{model}.pt")).to(device)

    input_path = os.path.join(input_dir, img)
    output_path = os.path.join(output_dir, img)

    _, pred = predict_image(input_path, model, device, patch_size)
    save_prediction(pred, input_path, output_path, threshold)


def main():
    args = get_args()
    assert all(
        os.path.exists(d) for d in [args.input_dir, args.output_dir, args.weights_dir]
    )

    regexp = r"L15\-\d{4}E\-\d{4}N\.tif"
    images = [f for f in os.listdir(args.input_dir) if re.match(regexp, f)]

    fn = partial(predict_and_save_image, **vars(args))

    with get_context("spawn").Pool(os.cpu_count() // 2) as pool:
        results = list(tqdm(pool.imap_unordered(fn, images), total=len(images)))

        all_successful = all(results)
        print(f"All tasks completed successfully: {all_successful}")

    print("Done.")


if __name__ == "__main__":
    main()
