import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
import torch
from typing import Literal
from vae.vae_architecture import VAE
import torchvision.transforms.v2 as v2

# VAE Preprocessing Transforms
vae_preprocess_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((256, 256)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        ),  # Scale to [-1,1] for Tanh activation in VAE decoder
    ]
)


def unNormalize(image, mean, std):
    return (image * std) + mean


def displayOriginal(image, mean=0.5, std=0.5):
    fig, ax = plt.subplots(layout="constrained")
    ax.imshow(np.transpose(unNormalize(image, mean, std), (1, 2, 0)))


def calc_scaling_factor(
    vae: VAE,
    dataset: torch.utils.data.DataLoader,
    device: Literal["cpu", "cuda", "mps"] = "cpu",
):
    latent_sum = latent_square_sum = latent_count = 0
    vae.to(device)
    vae.eval()
    with torch.inference_mode():
        for batch in dataset:
            batch = batch.to(device)

            mu, logvar = vae.encode(batch)
            z = vae.reparameterize(mu, logvar)
            z = z.to(torch.float64)

            latent_sum += z.sum().item()
            latent_square_sum += z.square().sum().item()
            latent_count += z.numel()

    latent_mean = latent_sum / latent_count
    latent_variance = (latent_square_sum / latent_count) - latent_mean**2
    latent_variance = max(latent_variance, 0)

    latent_std = latent_variance**0.5

    return 1 / latent_std


def kl_per_channel(mus: torch.Tensor, logvars: torch.Tensor):
    kl = -0.5 * (1 + logvars - mus.pow(2) - logvars.exp())

    # Only want channels
    kl = kl.sum(dim=(0, 2, 3))

    # Average over the number of samples
    kl /= mus.size(0)
    return kl


def setup_logger(
    NUM_HIDDEN_DIM: int,
    NUM_LATENT_DIM: int,
    LR: float,
    LOSS_BETA: int,
    filename: str | None = None,
) -> logging.Logger:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    if filename is None:
        filename = f"hidden_dim_{NUM_HIDDEN_DIM}_latent_dim_{NUM_LATENT_DIM}_lr_{LR}_beta_{LOSS_BETA}"
    logs_filename = f"logs/{timestamp}_{filename}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(logs_filename)],
    )
    logger = logging.getLogger("TrainingLogger")
    return logger
