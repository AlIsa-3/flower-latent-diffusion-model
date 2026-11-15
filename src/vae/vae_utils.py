import numpy as np
import matplotlib.pyplot as plt
import torch
from vae.vae_architecture import VAE
import torchvision.transforms.v2 as v2
from typing import Literal

# VAE Preprocessing Transforms
IMAGE_SIZE = 512
vae_preprocess_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
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


@torch.inference_mode()
def calc_shift_scale(
    vae: VAE,
    dataset: torch.utils.data.DataLoader,
    device: Literal["cuda", "mps", "cpu"],
    vae_channel_count: int,
):
    vae = vae.to(device)
    vae.eval()

    num_pixels = 0
    channel_sum = torch.zeros(vae_channel_count, device=device, dtype=torch.float32)
    channel_square_sum = torch.zeros(
        vae_channel_count, device=device, dtype=torch.float32
    )

    for batch in dataset:
        batch = batch.to(device)

        mu, logvar = vae.encode(batch)
        z = vae.reparameterize(mu, logvar)

        # mps does not support float64
        if device == "mps":
            z = z.to(torch.float32)
        else:
            z = z.to(torch.float64)

        # Sum over the channels
        channel_sum += z.sum(dim=(0, 2, 3))
        channel_square_sum += z.square().sum(dim=(0, 2, 3))
        # Total pixels
        num_pixels += z.shape[0] * z.shape[2] * z.shape[3]

    mean = channel_sum / num_pixels
    var = (channel_square_sum / num_pixels) - mean**2
    # Avoid sqrt n < 0
    std = torch.sqrt(torch.clamp(var, min=1e-12))

    shift = mean.float()
    scaling_factor = (1 / std).float()

    return shift, scaling_factor


def kl_per_channel(mus: torch.Tensor, logvars: torch.Tensor):
    kl = -0.5 * (1 + logvars - mus.pow(2) - logvars.exp())

    # Only want channels
    kl = kl.sum(dim=(0, 2, 3))

    # Average over the number of samples
    kl /= mus.size(0)
    return kl
