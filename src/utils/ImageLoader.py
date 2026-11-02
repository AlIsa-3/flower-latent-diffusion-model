import torch.utils.data as data
from PIL import Image
import pathlib


class FlattenedImageFolder(data.Dataset):
    def __init__(self, root_directory, transform=None) -> None:
        super().__init__()

        self.root_directory = pathlib.Path(root_directory)
        self.transform = transform

        # Some image types
        image_types = [".jpg", ".jpeg", ".png", ".webp"]

        self.image_paths = [
            self.root_directory.joinpath(file.name)
            for file in self.root_directory.rglob("*")
            if file.is_file() and file.suffix.lower() in image_types
        ]

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, index) -> Image.Image:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


    

        
