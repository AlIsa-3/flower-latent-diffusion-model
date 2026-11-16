import pathlib
import shutil
import os


def flatten_directory(
    path: pathlib.Path, new_path: pathlib.Path, file_extension: str = "jpg"
):
    """
    Create the new directory and move dataset into it, recursively searches directories

    Args:
        path: The original dataset path
        new_path: The destination directory for the flattened dataset
        file_extension: The file extension for the data which will be moved
    """
    if not os.path.isdir(new_path):
        os.makedirs(new_path)

    for path in path.rglob(f"*.{file_extension}"):
        if path.is_file():
            shutil.move(path, new_path.joinpath(path.name))


if __name__ == "__main__":
    # When running from the root directory --> src/setup/structure_dataset.py
    PATH = "oxford_flowers/dataset"
    flatten_directory(pathlib.Path(PATH), pathlib.Path("flattened_dataset"))
