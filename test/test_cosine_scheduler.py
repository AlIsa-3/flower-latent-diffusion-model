from unet.diffusion_utils import CosineNoiseScheduler
import torch

NUM_TIMESTEPS = 100
scheduler = CosineNoiseScheduler(num_timesteps=NUM_TIMESTEPS, s=0.008)

# Check that alpha_bars, alphas, betas all have same shape --> They should be 1D arrays


def test_alphas_shape():
    assert (
        scheduler.alphas.shape[0] == 100
    ), f"Alphas shape not 100, got {scheduler.alphas.shape[0]}"


def test_betas_shape():
    assert (
        scheduler.betas.shape[0] == 100
    ), f"Betas shape not 100, got {scheduler.betas.shape[0]}"


def test_alpha_bars_shape():
    assert (
        scheduler.alpha_bars.shape[0] == 100
    ), f"Alpha_bars shape not 100, got {scheduler.alphas_bars.shape}"


def test_betas_final():
    # Final beta should not be equal to 1
    assert (
        scheduler.betas[-1] != 1
    ), f"Final beta should not equal 1, got {scheduler.betas[-1]}"


def test_betas_initial():
    # Initial beta should not be 0 (that's why the offset s is there)
    assert (
        scheduler.betas[0] != 0
    ), f"First beta should not equal 0, got {scheduler.betas[0]}"


def test_add_noise():
    latent = torch.randn((1, 4, 32, 32))
    noise = torch.randn_like(latent)
    noised_latent = scheduler.add_noise(latent, noise, torch.tensor(5))
    assert (
        noised_latent.shape == latent.shape
    ), f"Adding noise shouldn't change latent shape. Latent shape = {latent.shape}, after noise, {noised_latent.shape}"


def test_forward_diffusion_alpha_bar_zero():
    alpha_bar = torch.tensor(0).reshape(-1, 1, 1, 1)
    latent = torch.ones((1, 4, 32, 32))
    noise = torch.randn_like(latent)
    noised_latent = scheduler._forward_diffusion(latent, alpha_bar, noise)
    assert (
        noised_latent == noise
    ).all(), f"with alpha_bar 0, the noised latent should equal the noise"


def test_forward_diffusion_alpha_bar_1():
    # Should get the original latent
    alpha_bar = torch.tensor(1).reshape(-1, 1, 1, 1)
    latent = torch.ones((1, 4, 32, 32))
    noise = torch.randn_like(latent)
    noised_latent = scheduler._forward_diffusion(latent, alpha_bar, noise)
    assert (
        noised_latent == latent
    ).all(), f"With alpha_bar 1, the noised latent should equal the original latent"
