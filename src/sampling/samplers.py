import torch
from sampling.sampler_utils import get_ancestral_step


def euler_ancestral_update(
    latent: torch.Tensor,
    derivative: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
):
    sigma_down, sigma_up = get_ancestral_step(sigma, sigma_next)

    next_latent = euler_update(latent, derivative, sigma, sigma_down)

    next_latent += torch.randn_like(next_latent) * sigma_up
    return next_latent


def euler_update(
    latent: torch.Tensor,
    derivative: torch.Tensor,
    sigma: torch.Tensor,
    sigma_target: torch.Tensor,
):
    return latent + derivative * (sigma_target - sigma)
