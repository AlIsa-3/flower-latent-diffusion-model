import torch
import torch.nn as nn
import numpy as np
from sampling.sampler_utils import calc_sigmas


class CosineNoiseScheduler(nn.Module):
    # Types to avoid typechecking mismatch
    alpha_bars: torch.Tensor
    alphas: torch.Tensor
    betas: torch.Tensor
    sigmas: torch.Tensor

    def __init__(
        self, num_timesteps: int, s: float = 0.008, zero_terminal_snr: bool = False
    ) -> None:
        super().__init__()
        self.zero_terminal_snr = zero_terminal_snr
        self.num_timesteps = num_timesteps
        self.s = s

        steps = torch.arange(num_timesteps + 1, dtype=torch.float32)

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

        if self.zero_terminal_snr:
            alpha_bars = self.rescale_zero_terminal_snr(alpha_bars)

            # Reconstruct individual alphas and betas
            alphas = torch.cat(
                [
                    alpha_bars[:1],
                    alpha_bars[1:] / alpha_bars[:-1],
                ]
            )

            betas = 1 - alphas
        sigmas = calc_sigmas(alpha_bars)

        # Register so saved with model
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)

    def _forward_diffusion(
        self, z_0: torch.Tensor, alpha_bar: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        return (torch.sqrt(alpha_bar) * z_0) + (torch.sqrt(1 - alpha_bar) * eps)

    def add_noise(
        self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        # Add noise to latent batch according to their timesteps
        alpha_bar = self.alpha_bars[timesteps]  # Get corresponding alpha bar values
        alpha_bar = alpha_bar.reshape(-1, 1, 1, 1)  # reshaped to match latents

        # Apply forward diffusion equation
        noisy_latents = self._forward_diffusion(latents, alpha_bar, noise)

        return noisy_latents

    def rescale_zero_terminal_snr(
        self,
        alpha_bars: torch.Tensor,
    ) -> torch.Tensor:
        alphas_bar_sqrt = torch.sqrt(alpha_bars)

        # Preserve the first timestep
        first = alphas_bar_sqrt[0].clone()

        # Shift the final timestep to zero
        last = alphas_bar_sqrt[-1].clone()

        alphas_bar_sqrt = alphas_bar_sqrt - last

        # Rescale to preserve the original first timestep
        alphas_bar_sqrt = alphas_bar_sqrt * first / (first - last)

        return alphas_bar_sqrt.square()


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


def calc_velocity(
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: CosineNoiseScheduler,
) -> torch.Tensor:
    alpha_bar = scheduler.alpha_bars[timesteps]
    alpha_bar = alpha_bar.reshape(-1, 1, 1, 1)

    return torch.sqrt(alpha_bar) * noise - torch.sqrt(1 - alpha_bar) * latents


def min_snr_weighting(
    timesteps: torch.Tensor,
    scheduler: CosineNoiseScheduler,
    gamma: float = 5.0,
) -> torch.Tensor:
    if gamma <= 0:
        raise ValueError("gamma must be greater than 0")

    alpha_bar = scheduler.alpha_bars[timesteps]

    # Avoid division by 0
    snr = alpha_bar / torch.clamp((1.0 - alpha_bar), min=1e-12)

    weights = torch.clamp(snr, max=gamma) / (snr + 1.0)

    return weights
