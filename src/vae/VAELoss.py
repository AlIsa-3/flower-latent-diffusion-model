import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Literal


class VAELoss(nn.Module):
    """
    VAE Loss function with optional beta parameter to allow creation of a beta-VAE
    """

    def __init__(
        self, beta: float = 1, reconstruction_loss: Literal["mse", "mae"] = "mse"
    ) -> None:
        super().__init__()
        self.beta = beta

        self.reconstruction_loss = reconstruction_loss

    def _KL_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - (mu**2) - torch.exp(logvar)) / mu.shape[0]

    def _reconstruction_loss(
        self, reconstruction: torch.Tensor, original: torch.Tensor
    ) -> torch.Tensor:
        if self.reconstruction_loss == "mse":
            return (
                F.mse_loss(reconstruction, original, reduction="sum")
                / original.shape[0]
            )
        else:
            return (
                F.l1_loss(reconstruction, original, reduction="sum") / original.shape[0]
            )

    def forward(
        self,
        vae_output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        original_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reconstruction, mu, logvar = vae_output

        reconstruction_loss = self._reconstruction_loss(reconstruction, original_input)
        kl_divergence = self._KL_divergence(mu, logvar)

        total_loss = reconstruction_loss + (self.beta * kl_divergence)

        return total_loss, kl_divergence, reconstruction_loss
