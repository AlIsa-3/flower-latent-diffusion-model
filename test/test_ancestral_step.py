from sampling.samplers import get_ancestral_step
import torch

sigma = torch.tensor(4)
sigma_next = torch.tensor(2)


def test_get_ancestral_step_sigma_up():
    sigma_down, sigma_up = get_ancestral_step(sigma, sigma_next)
    assert torch.allclose(
        sigma_up, torch.sqrt(torch.tensor(3))
    ), f"With sigma = {sigma.item()} and sigma_next = {sigma_next.item()}, sigma_up should equal sqrt(3), got {sigma_up.item()}"


def test_get_ancestral_step_sigma_down():
    sigma_down, sigma_up = get_ancestral_step(sigma, sigma_next)
    assert torch.allclose(
        sigma_down, torch.tensor(1.0)
    ), f"With sigma = {sigma.item()} and sigma_next = {sigma_next.item()}, sigma_down should equal 1, got {sigma_down.item()}"
