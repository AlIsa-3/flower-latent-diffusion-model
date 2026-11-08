from unet.unet_architecture import UNet
import torch

input_latent_channels = 8
hidden_dimensions = 16

model = UNet(input_latent_channels, hidden_dimensions)


def test_unet_output_dimensions():
    # B,C,H,W
    input_latent = torch.randn((1, 8, 32, 32))
    time_step = torch.randint(10, size=(1,))
    assert model(input_latent, time_step).shape == (1, 8, 32, 32)
