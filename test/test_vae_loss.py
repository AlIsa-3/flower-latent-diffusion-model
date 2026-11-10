from vae.VAELoss import VAELoss
import torch
import pytest


# Standard VAE Loss
@pytest.fixture()
def loss():
    return VAELoss(beta=1)


generator = torch.manual_seed(453)

# Values need to be an array for the loss function to work
mu = torch.tensor(0).unsqueeze(0)
var = torch.tensor(1).unsqueeze(0)
logvar = torch.log(var)

original = torch.randn((1, 3, 256, 256), generator=generator)
reconstruction = original.clone()


def test_KL_divergence(loss):
    assert (
        loss._KL_divergence(mu, logvar) == 0
    ), "KL Divergence should be 0 when mu and logvar are normal"


def test_KL_divergence_non_normal(loss):
    assert (
        loss._KL_divergence(torch.tensor(3).unsqueeze(0), torch.tensor(1).unsqueeze(0))
        != 0
    ), f"KL Divergence should not be 0 with non normal mu, logvar. Got: {mu = }, {logvar = }"


def test_reconstruction_loss_identical(loss):
    assert (
        loss._reconstruction_loss(reconstruction, original) == 0
    ), "Reconstruction loss should be 0 with identical reconstruction"


def test_reconstruction_loss(loss):
    assert (
        loss._reconstruction_loss(
            torch.randn_like(original, generator=generator), original
        )
        != 0
    ), "Reconstruction loss should not be 0 with non identical reconstruction"


def test_vae_loss_kl(loss):
    # Identical reconstruction but non normal mu and logvar
    mu = torch.tensor(5).unsqueeze(0)
    logvar = torch.tensor(5).unsqueeze(0)
    assert (
        loss((reconstruction, mu, logvar), original) != 0
    ), f"VAE Loss should not be 0 with non normal mu and logvar {mu = },{logvar = }"


def test_vae_loss_reconstruction(loss):
    # Normal mu and logvar but non identical reconstruction
    assert (
        loss((torch.randn_like(original, generator=generator), mu, logvar), original)
        != 0
    ), "VAE loss should not be 0 with non identical reconstruction"
