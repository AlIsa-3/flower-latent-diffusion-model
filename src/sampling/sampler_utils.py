from __future__ import annotations
import torch
from typing import Literal, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from unet.unet_architecture import UNet


def to_derivative(
    latent: torch.Tensor,
    denoised_latent: torch.Tensor,
    sigma: torch.Tensor,
    min_eps: float = 1e-12,
):
    sigma = torch.clamp(sigma, min=min_eps)
    return (latent - denoised_latent) / sigma


def calc_sigmas(alpha_bars: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(((1 - alpha_bars) / alpha_bars))


def estimate_clean_vp_latent_from_eps(
    latent: torch.Tensor, eps: torch.Tensor, alpha_bar: torch.Tensor
) -> torch.Tensor:
    return (latent - (torch.sqrt(1 - alpha_bar) * eps)) / torch.sqrt(alpha_bar)


def calc_denoised_latent(
    latent: torch.Tensor, sigma: torch.Tensor, eps: torch.Tensor
) -> torch.Tensor:
    return latent - sigma * eps


def get_ancestral_step(
    sigma: torch.Tensor, sigma_next: torch.Tensor, clamp_min: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    # Avoid division by 0
    sigma = torch.clamp(sigma, min=clamp_min)
    # Clamp to avoid multiplying by negative in sqrt
    sigma_up = torch.sqrt(
        (sigma_next**2 * torch.clamp((sigma**2 - sigma_next**2), min=0)) / sigma**2
    )

    sigma_down = sigma_next**2 / sigma

    return sigma_down, sigma_up


def convert_from_sigmas(x_sigma: torch.Tensor, sigma: torch.Tensor):
    """
    Converts from a sigma-space latent to a VP latent
    """
    return x_sigma / torch.sqrt(1 + sigma**2)


@torch.inference_mode()
def sampling_loop(
    unet: UNet,
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    latent: torch.Tensor,
    sampler: Callable,
    device: Literal["cuda", "mps", "cpu"],
    prediction_type: Literal["eps", "v"],
):
    if prediction_type not in ("eps", "v"):
        raise ValueError("prediction_type must be 'eps' or 'v'")

    if sigmas.ndim != 1 or timesteps.ndim != 1:
        raise ValueError("sigmas and timesteps must be 1D")

    if sigmas.numel() != timesteps.numel() + 1:
        raise ValueError("sigmas must contain exactly one more element than timesteps")

    unet = unet.to(device)
    unet.eval()

    sigmas = sigmas.to(device)
    timesteps = timesteps.to(device)
    latent = latent.to(device) * sigmas[0]

    for idx, timestep in enumerate(timesteps):
        timestep = timestep.expand(latent.shape[0])

        sigma = sigmas[idx]
        sigma_next = sigmas[idx + 1]

        vp_latent = convert_from_sigmas(latent, sigma)

        prediction = unet(vp_latent, timestep)

        if prediction_type == "eps":
            denoised_latent = calc_denoised_latent(latent, sigma, prediction)

        elif prediction_type == "v":
            alpha_bar = 1 / (1 + sigma.square())

            denoised_latent = (
                torch.sqrt(alpha_bar) * vp_latent
                - torch.sqrt(1 - alpha_bar) * prediction
            )

        derivative = to_derivative(latent, denoised_latent, sigma)

        latent = sampler(latent, derivative, sigma, sigma_next)

    return latent / unet.latent_scaling_factor
