import pytest
import torch
from unet.diffusion_utils import CosineNoiseScheduler, calc_velocity


@pytest.fixture
def scheduler():
    return CosineNoiseScheduler(num_timesteps=1000)


def test_calc_velocity_shape(scheduler):
    batch_size = 4
    latents = torch.randn(batch_size, 8, 32, 32)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([0, 100, 500, 999])

    velocity = calc_velocity(latents, noise, timesteps, scheduler)

    assert velocity.shape == latents.shape


def test_calc_velocity_values(scheduler):
    latents = torch.randn(4, 8, 32, 32)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([0, 100, 500, 999])

    velocity = calc_velocity(latents, noise, timesteps, scheduler)

    alpha_bar = scheduler.alpha_bars[timesteps]
    alpha_bar = alpha_bar.reshape(-1, 1, 1, 1)

    expected = torch.sqrt(alpha_bar) * noise - torch.sqrt(1 - alpha_bar) * latents

    assert torch.allclose(velocity, expected)


def test_calc_velocity_timestep_zero(scheduler):
    latents = torch.randn(2, 8, 32, 32)
    noise = torch.randn_like(latents)
    timesteps = torch.zeros(2, dtype=torch.long)

    velocity = calc_velocity(latents, noise, timesteps, scheduler)

    alpha_bar = scheduler.alpha_bars[timesteps]
    alpha_bar = alpha_bar.reshape(-1, 1, 1, 1)

    expected = torch.sqrt(alpha_bar) * noise - torch.sqrt(1 - alpha_bar) * latents

    assert torch.allclose(velocity, expected)


def test_calc_velocity_invalid_batch_size(scheduler):
    latents = torch.randn(4, 8, 32, 32)
    noise = torch.randn(3, 8, 32, 32)
    timesteps = torch.tensor([0, 100, 500, 999])

    with pytest.raises(RuntimeError):
        calc_velocity(latents, noise, timesteps, scheduler)


def test_calc_velocity_invalid_timestep(scheduler):
    latents = torch.randn(2, 8, 32, 32)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([0, 1000])

    with pytest.raises(IndexError):
        calc_velocity(latents, noise, timesteps, scheduler)


def test_velocity_recovers_clean_latents(scheduler):
    latents = torch.randn(4, 8, 32, 32)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([10, 100, 500, 900])

    noised_latents = scheduler.add_noise(latents, noise, timesteps)

    velocity = calc_velocity(latents, noise, timesteps, scheduler)

    alpha_bar = scheduler.alpha_bars[timesteps]
    alpha_bar = alpha_bar.reshape(-1, 1, 1, 1)

    recovered_latents = (
        torch.sqrt(alpha_bar) * noised_latents - torch.sqrt(1 - alpha_bar) * velocity
    )

    assert torch.allclose(recovered_latents, latents, rtol=1e-4, atol=1e-5)
