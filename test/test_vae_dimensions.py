from vae.vae_architecture import VAE
import torch
import pytest

IMAGE_SIZE = 512
DOWNSCALE_FACTOR = 8
BOTTLENECK_SIZE = IMAGE_SIZE / DOWNSCALE_FACTOR

input_channels = 3
hidden_dimensions = 16
LATENT_DIMENSIONS = 4


@pytest.fixture()
def model():
    return VAE(
        input_channels=input_channels,
        hidden_dimensions=hidden_dimensions,
        latent_dimensions=LATENT_DIMENSIONS,
    )


def test_vae_reconstruction_dimensions(model):
    model_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    # Get model output
    reconstruction, mu, logvar = model(model_input)

    assert (
        reconstruction.shape == model_input.shape
    ), "Reconstruction does not match input dimensions"
    assert (
        mu.shape == logvar.shape
    ), f"Mu and logvar don't have the same shape| mu: {mu.shape}, logvar: {logvar.shape}"
    assert (
        mu.shape
        == (
            1,
            LATENT_DIMENSIONS,
            BOTTLENECK_SIZE,
            BOTTLENECK_SIZE,
        )
    ), f"Mu shape is not (1,{LATENT_DIMENSIONS},{BOTTLENECK_SIZE},{BOTTLENECK_SIZE}), got {mu.shape}"


def test_vae_mu_dimensions(model):
    model_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    # Get model output
    reconstruction, mu, logvar = model(model_input)
    assert (
        mu.shape
        == (
            1,
            LATENT_DIMENSIONS,
            BOTTLENECK_SIZE,
            BOTTLENECK_SIZE,
        )
    ), f"Mu shape is not (1,{LATENT_DIMENSIONS},{BOTTLENECK_SIZE},{BOTTLENECK_SIZE}), got {mu.shape}"


def test_vae_logvar_dimensions(model):
    model_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    # Get model output
    reconstruction, mu, logvar = model(model_input)
    assert (
        logvar.shape
        == (
            1,
            LATENT_DIMENSIONS,
            BOTTLENECK_SIZE,
            BOTTLENECK_SIZE,
        )
    ), f"logvar shape is not (1,{LATENT_DIMENSIONS},{BOTTLENECK_SIZE},{BOTTLENECK_SIZE}), got {logvar.shape}"


def test_vae_mu_logvar_dimensions_equal(model):
    model_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    # Get model output
    reconstruction, mu, logvar = model(model_input)
    assert (
        mu.shape == logvar.shape
    ), f"Mu and logvar don't have the same shape| mu: {mu.shape}, logvar: {logvar.shape}"


if __name__ == "__main__":
    pass
