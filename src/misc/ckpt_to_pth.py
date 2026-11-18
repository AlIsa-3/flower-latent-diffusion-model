import torch
import pathlib
import os
import argparse


def convert_to_pth(
    ckpt_path: os.PathLike, destination_path: os.PathLike | None = None
) -> None:
    """
    Convert a .ckpt model training checkpoint into a .pth file containing the model's weights

    Args:
        ckpt_path: the path to the model checkpoint
        destination_path: the path where the model will be saved. Defaults to the same path, with a differnet extension
    Returns:
        No return, but saves the converted weights to disk at destination_path
    """

    ckpt_path = pathlib.Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if destination_path is None:
        destination_path = ckpt_path.with_suffix(".pth")
    else:
        destination_path = pathlib.Path(destination_path)

    # Ensure that destination path has the correct extension
    if destination_path.suffix != ".pth":
        destination_path = destination_path.with_suffix(".pth")
        # Create destination folders if not present
        destination_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt["model_state_dict"], destination_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", help="The path to the .ckpt file")
    parser.add_argument(
        "--destination",
        "-d",
        required=False,
        help="The destination for the converted file",
        default=None,
    )

    args = parser.parse_args()

    try:
        convert_to_pth(ckpt_path=args.ckpt_path, destination_path=args.destination)
    except FileNotFoundError:
        print(f"Path to ckpt was invalid, received: {args.ckpt_path}")
        print("Ensure that you include the file")
