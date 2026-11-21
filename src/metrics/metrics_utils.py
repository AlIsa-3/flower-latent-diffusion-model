from __future__ import annotations
import torch
import torch.utils.data as data
import torchvision.transforms.v2 as v2
from utils.ImageLoader import FlattenedImageFolder
from typing import Literal, TYPE_CHECKING
import os

if TYPE_CHECKING:
    from inference.inference import Inference

IMAGE_SIZE = 256
test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


def generate_unet_test_set(
    inference: Inference,
    count: int,
    prediction_type: Literal["eps", "v"],
    batch_size: int = 8,
    transforms: v2.Compose = test_transforms,
    regenerate: bool = False,
) -> data.DataLoader:
    # If directory already exists then don't generate
    if not os.path.exists("test_generated_images") or regenerate:
        starting_step = (
            inference.scheduler.num_timesteps
            if prediction_type == "v"
            else inference.scheduler.num_timesteps - 100
        )
        # Use 0-count-1 as the seed for reproducibility
        for i in range(count):
            if (i + 1) % 50 == 0:
                print(f"{i+1} / {count}")
            inference.sample(
                steps=25,
                seed=i,
                latent_size=(16, 32, 32),
                image_name=f"test_generated_images/image_{i}",
                starting_timestep=starting_step,
            )
        print("Generated Test Set")
    # Load Images
    dataset = FlattenedImageFolder("test_generated_images", transform=transforms)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_real_test_set(
    generator_seed: int = 423,
    batch_size: int = 8,
    transforms: v2.Compose = test_transforms,
) -> data.DataLoader:
    gen = torch.Generator().manual_seed(generator_seed)
    dataset = FlattenedImageFolder("flattened_dataset", transform=transforms)
    _, _, test = data.random_split(dataset, [0.8, 0.1, 0.1], generator=gen)
    return data.DataLoader(test, batch_size=batch_size, shuffle=False)
