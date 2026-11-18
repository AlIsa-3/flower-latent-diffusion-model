from unet.diffusion_utils import CosineNoiseScheduler
import torch
import pytest

NUM_TIMESTEPS = 100


@pytest.fixture
def scheduler():
    return CosineNoiseScheduler(num_timesteps=NUM_TIMESTEPS, s=0.008)


@pytest.fixture
def zsnr_scheduler():
    return CosineNoiseScheduler(
        num_timesteps=NUM_TIMESTEPS, s=0.008, zero_terminal_snr=True
    )


# Check that alpha_bars, alphas, betas all have same shape --> They should be 1D arrays


def test_alphas_shape(scheduler):
    assert (
        scheduler.alphas.shape[0] == 100
    ), f"Alphas shape not 100, got {scheduler.alphas.shape[0]}"


def test_betas_shape(scheduler):
    assert (
        scheduler.betas.shape[0] == 100
    ), f"Betas shape not 100, got {scheduler.betas.shape[0]}"


def test_alpha_bars_shape(scheduler):
    assert (
        scheduler.alpha_bars.shape[0] == 100
    ), f"Alpha_bars shape not 100, got {scheduler.alpha_bars.shape}"


def test_betas_final(scheduler):
    # Final beta should not be equal to 1
    assert (
        scheduler.betas[-1] != 1
    ), f"Final beta should not equal 1, got {scheduler.betas[-1]}"


def test_betas_initial(scheduler):
    # Initial beta should not be 0 (that's why the offset s is there)
    assert (
        scheduler.betas[0] != 0
    ), f"First beta should not equal 0, got {scheduler.betas[0]}"


def test_add_noise(scheduler):
    latent = torch.randn((1, 4, 32, 32))
    noise = torch.randn_like(latent)
    noised_latent = scheduler.add_noise(latent, noise, torch.tensor(5))
    assert (
        noised_latent.shape == latent.shape
    ), f"Adding noise shouldn't change latent shape. Latent shape = {latent.shape}, after noise, {noised_latent.shape}"


def test_forward_diffusion_alpha_bar_zero(scheduler):
    alpha_bar = torch.tensor(0).reshape(-1, 1, 1, 1)
    latent = torch.ones((1, 4, 32, 32))
    noise = torch.randn_like(latent)
    noised_latent = scheduler._forward_diffusion(latent, alpha_bar, noise)
    assert (
        noised_latent == noise
    ).all(), "with alpha_bar 0, the noised latent should equal the noise"


def test_forward_diffusion_alpha_bar_1(scheduler):
    # Should get the original latent
    alpha_bar = torch.tensor(1).reshape(-1, 1, 1, 1)
    latent = torch.ones((1, 4, 32, 32))
    noise = torch.randn_like(latent)
    noised_latent = scheduler._forward_diffusion(latent, alpha_bar, noise)
    assert (
        noised_latent == latent
    ).all(), "With alpha_bar 1, the noised latent should equal the original latent"


def test_forward_diffusion(scheduler):
    latent = torch.ones((1, 4, 32, 32))
    noise = torch.ones_like(latent)
    alpha_bar = torch.tensor(0.25)
    noised_latent = scheduler._forward_diffusion(latent, alpha_bar, noise)
    expected = torch.full_like(latent, 1.3660254)
    assert torch.allclose(noised_latent, expected, atol=1e-6)


def test_zero_snr_final_alpha_bar(zsnr_scheduler):
    assert zsnr_scheduler.alpha_bars[-1].item() == 0


def test_zero_snr_final_alpha(zsnr_scheduler):
    assert zsnr_scheduler.alphas[-1].item() == 0


def test_zero_snr_final_beta(zsnr_scheduler):
    assert zsnr_scheduler.betas[-1].item() == 1


def test_zero_snr_preserves_initial_alpha_bar(scheduler, zsnr_scheduler):
    assert torch.allclose(
        scheduler.alpha_bars[0],
        zsnr_scheduler.alpha_bars[0],
    )


def test_zero_snr_schedule_is_finite(zsnr_scheduler):
    assert torch.isfinite(zsnr_scheduler.alpha_bars).all()
    assert torch.isfinite(zsnr_scheduler.alphas).all()
    assert torch.isfinite(zsnr_scheduler.betas).all()
    assert torch.isfinite(zsnr_scheduler.sigmas).all()


def test_zero_snr_terminal_add_noise(zsnr_scheduler):
    latent = torch.randn((2, 4, 32, 32))
    noise = torch.randn_like(latent)

    timesteps = torch.full(
        (2,),
        zsnr_scheduler.num_timesteps - 1,
        dtype=torch.long,
    )

    result = zsnr_scheduler.add_noise(latent, noise, timesteps)

    torch.testing.assert_close(result, noise)
