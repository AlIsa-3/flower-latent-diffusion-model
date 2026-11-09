import matplotlib.pyplot as plt
import numpy as np
import torch
import pathlib
from typing import Literal


from vae.vae_utils import unNormalize


def visualize_vae(
    input_image,
    reconstruction,
    figure_name: str,
    figure_directory: str | pathlib.Path,
    ext: str = "png",
):
    figure_directory = pathlib.Path(figure_directory)
    figure_directory.mkdir(parents=True, exist_ok=True)

    input_image = unNormalize(np.transpose(input_image, (1, 2, 0)), mean=0.5, std=0.5)
    reconstruction = unNormalize(
        np.transpose(reconstruction, (1, 2, 0)), mean=0.5, std=0.5
    )

    fig, ax = plt.subplots(nrows=2, figsize=(12, 24), layout="constrained")
    ax[0].imshow(input_image)
    ax[0].set_title("Original")
    ax[1].imshow(reconstruction)
    ax[1].set_title("Reconstruction")

    fig.suptitle(figure_name)
    figure_path = figure_directory / f"{figure_name}.{ext}"
    fig.savefig(str(figure_path), dpi=100)


def visualize_vae_generations(
    vae,
    vae_latent_shape: tuple[int, int, int],
    figure_filename: str,
    ext: str = "png",
    device: Literal["cuda", "mps", "cpu"] = "cpu",
):
    vae = vae.to(device)
    fig, ax = plt.subplots(nrows=20, ncols=20, figsize=(32, 32), layout="constrained")
    with torch.inference_mode():
        for row in range(20):
            for column in range(20):
                noise = torch.randn(vae_latent_shape).unsqueeze(0).to(device)
                generation = vae.decode(noise).squeeze(0).cpu()
                ax[row, column].imshow(
                    unNormalize(np.transpose(generation, (1, 2, 0)), 0.5, 0.5)
                )
                ax[row, column].axis("off")
    fig.savefig(f"{figure_filename}.{ext}")
    plt.close(fig)
