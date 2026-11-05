import numpy as np
import matplotlib.pyplot as plt
import pathlib
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
    VAE: VAE,
    dataset: torch.utils.data.DataLoader,
    device: Literal["cpu", "cuda", "mps"] = "cpu",
):
    latent_sum = latent_square_sum = latent_count = 0

    VAE.eval()
    with torch.inference_mode():
        for batch in dataset:
            batch = batch.to(device)

            mu, logvar = VAE.encode(batch)
            z = VAE.reparameterize(mu, logvar)

            latent_sum += z.sum().item()
            latent_square_sum += z.square().sum().item()
            latent_count += z.numel()

    latent_mean = latent_sum / latent_count
    latent_variance = (latent_square_sum / latent_count) - latent_mean**2

    latent_std = latent_variance**0.5

    return 1 / latent_std


class EarlyStop:
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 10,
        checkpoint_path: str = "best_model.ckpt",
    ):
        if patience < 1:
            raise ValueError("patience must be >= 1")

        if min_delta < 0:
            raise ValueError("min_delta must be >= 0")

        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = pathlib.Path(checkpoint_path)

        self.best_metric = float("inf")

        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False

    def _is_improvement(self, metric: float) -> bool:
        return metric < self.best_metric - self.min_delta

    def __call__(
        self,
        metric: float,
        model: torch.nn.Module,
        epoch: int,
        optimizer=None,
    ) -> bool:
        # Avoid NaN or infinite metrics
        if not torch.isfinite(torch.tensor(metric)):
            raise ValueError(f"Invalid metric: {metric}")

        # If improvement then reset early stop counter
        if self._is_improvement(metric):
            self.best_metric = metric
            self.best_epoch = epoch
            self.counter = 0

            checkpoint = {
                "epoch": epoch,
                "metric": metric,
                "model_state_dict": model.state_dict(),
            }

            if optimizer is not None:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()

            torch.save(checkpoint, self.checkpoint_path)

        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

    def status(self):
        return {
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "patience": self.patience,
            "early_stop": self.early_stop,
        }


def kl_per_channel(mus, logvars):
    kl = -0.5 * (1 + logvars - mus.pow(2) - logvars.exp())

    # Only want channels
    kl = kl.sum(dim=(0, 2, 3))

    # Average over the number of samples
    kl /= mus.size(0)
    return kl
