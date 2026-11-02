import torch.nn as nn
import torch.nn.functional as F
import torch


class VAELoss(nn.Module):
    """
    VAE Loss function with optional beta parameter to allow creation of a beta-VAE
    """

    def __init__(self, beta: float = 1) -> None:
        super().__init__()
        self.beta = beta

    def _KL_divergence(self, mu, logvar):
        return 0.5 * torch.sum(1 + logvar - (mu**2) - torch.exp(logvar))

    def _reconstruction_loss(self, reconstruction, original):
        return F.mse_loss(reconstruction, original, reduction="sum")

    def forward(self, vae_output: tuple, original_input):
        reconstruction, mu, logvar = vae_output

        reconstruction_loss = self._reconstruction_loss(reconstruction, original_input)
        kl_divergence = self._KL_divergence(mu, logvar)

        #       Reconstruction                    KL Divergence averaged over the batch
        return reconstruction_loss + self.beta * (
            kl_divergence / original_input.shape[0]
        )
