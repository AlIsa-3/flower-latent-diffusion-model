import torch
import torch.nn as nn
import numpy as np


class CosineNoiseScheduler(nn.Module):
    # Types to avoid typechecking mismatch
    alpha_bars: torch.Tensor
    alphas: torch.Tensor
    betas: torch.Tensor

    def __init__(self, num_timesteps: int, s: float = 0.008) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.s = s

        steps = torch.arange(self.num_timesteps + 1, dtype=torch.float32)

        f = (
            torch.cos(
                (((steps / self.num_timesteps) + self.s) / (1 + s)) * (torch.pi / 2)
            )
            ** 2
        )

        alpha_bars = f / f[0]
        # Need to divide each cumulative alpha by the previous one
        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])

        # Clamp betas to avoid case where beta == 1 --> alpha == 0
        betas = torch.clamp(betas, 0, 0.999)

        alphas = 1 - betas

        # Recalculate alpha_bars based on the new alphas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # Register so saved with model
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)

    def _forward_diffusion(
        self, z_0: torch.Tensor, alpha_bar: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        return ((alpha_bar**0.5) * z_0) + (eps * ((1 - alpha_bar) ** 2))

    def add_noise(
        self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        # Add noise to latent batch according to their timesteps
        alpha_bar = self.alpha_bars[timesteps]  # Get corresponding alpha bar values
        alpha_bar = alpha_bar.reshape(-1, 1, 1, 1)  # reshaped to match latents

        # Apply forward diffusion equation
        noisy_latents = self._forward_diffusion(latents, alpha_bar, noise)

        return noisy_latents


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, embedding_dimensions: int = 64):
        super().__init__()
        self.embedding_dimensions = embedding_dimensions

    def forward(self, timesteps, FREQUENCY_RANGE: int = 10000):
        # Half because have both sin and cos
        half_dimensions = self.embedding_dimensions // 2

        # Calculate frequencies to use --> can model time changes better
        frequencies = torch.exp(
            -np.log(FREQUENCY_RANGE)
            * torch.arange(
                half_dimensions, device=timesteps.device, dtype=torch.float32
            )
            / half_dimensions
        )

        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)

        embeddings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        return embeddings
