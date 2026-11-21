from PIL import Image
import pathlib


def shrink_image(path, isFolder=False) -> None:
    path = pathlib.Path(path)
    if isFolder:
        images = [
            image
            for image in path.rglob("*")
            if image.suffix in [".png", ".jpg", ".webp", ".svg"]
        ]
        print(f"Found {len(images)} Images")
        for image_path in images:
            print(f"Working on {image_path}")
            image = Image.open(image_path)
            image.save(
                image_path.with_suffix(".webp"), format="WEBP", quality=85, method=6
            )
    else:
        image = Image.open(path)
        image.save(path.with_suffix(".webp"), format="WEBP", quality=85, method=6)


if __name__ == "__main__":
    shrink_image("training_images/", isFolder=True)
