import argparse
import os
import re

from tqdm import tqdm

from models import get_all_models, make_model
from utils.misc import get_device
from utils.predictions import predict_image, save_prediction


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

    return parser.parse_args()


def main():
    args = get_args()
    assert all(
        os.path.exists(d) for d in [args.input_dir, args.output_dir, args.weights_dir]
    )
    device = get_device()

    model = (
        make_model(args.model)
        .load(os.path.join(args.weights_dir, f"{args.model}.pt"))
        .to(device)
    )

    regexp = r"L15\-\d{4}E\-\d{4}N\.tif"
    images = [f for f in os.listdir(args.input_dir) if re.match(regexp, f)]

    for img in tqdm(images):
        save_prediction(
            predict_image(
                model,
                device,
                os.path.join(args.input_dir, img),
                args.patch_size,
            )[1],
            os.path.join(args.output_dir, img),
        )

    print("Done.")


if __name__ == "__main__":
    main()
