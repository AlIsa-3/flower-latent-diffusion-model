from sampling.sampler_utils import convert_from_sigmas
import torch


sigma = torch.tensor(1.0)
x_sigma = torch.tensor(4.0)


def test_convert_from_sigmas():
    result = convert_from_sigmas(x_sigma, sigma)
    assert torch.isclose(
        result, 2 * torch.sqrt(torch.tensor(2))
    ), f"With sigma = 1 and x_sigma = 4, converting from sigmas should give 2*sqrt(2). Got {result}"
