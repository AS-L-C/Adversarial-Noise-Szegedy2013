import argparse

from datasets import load_dataset
from PIL import Image


def download_and_save_images(
    store_path,
    n_images,
    split,
):
    dataset = load_dataset(
        "imagenet-1k",
        split=split,
        streaming=True,
        trust_remote_code=True,
    ).take(n_images)

    for ind, img_dict in enumerate(dataset):
        img, label = img_dict["image"], img_dict["label"]
        img.save(f"{store_path}/{split}_{ind}_label_{label}.jpg")
    return


def parse_arguments(parser):
    parser.add_argument(
        "--store_path",
        default="../Original_Images",
        help="Path to folder that will contain the downloaded images",
    )

    parser.add_argument(
        "--n_images",
        default=30,
        help="The number of images to download",
        type=int,
    )

    parser.add_argument(
        "--split",
        default="validation",
        help="The Imagenet split to select images from",
    )

    args = parser.parse_args()
    return vars(args)


def main():
    parser = argparse.ArgumentParser(description="Download and store ImageNet images.")
    args = parse_arguments(parser)
    print(f"Download to folder '{args['store_path']}' in progress...")
    download_and_save_images(**args)
    print("Download completed.")


if __name__ == "__main__":
    main()
