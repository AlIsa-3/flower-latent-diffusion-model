import torch
import torch.nn as nn
from utils.ResidualBlock import ResidualBlock


class VAE(nn.Module):
    def __init__(
        self, input_channels: int, hidden_dimensions: int, latent_dimensions: int = 4
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=hidden_dimensions,
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dimensions),
            nn.SiLU(),
            ResidualBlock(hidden_dimensions),
            ResidualBlock(hidden_dimensions),
            # 128x128
            nn.Conv2d(
                in_channels=hidden_dimensions,
                out_channels=hidden_dimensions * 2,
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dimensions * 2),
            nn.SiLU(),
            ResidualBlock(hidden_dimensions * 2),
            ResidualBlock(hidden_dimensions * 2),
            # 64x64
            nn.Conv2d(
                in_channels=hidden_dimensions * 2,
                out_channels=hidden_dimensions * 4,
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dimensions * 4),
            nn.SiLU(),
            ResidualBlock(hidden_dimensions * 4),
            ResidualBlock(hidden_dimensions * 4),
            # 32x32
        )
        # Convert Encoded Data to mu and logvar
        self.to_mu = nn.Conv2d(
            4 * hidden_dimensions, latent_dimensions, kernel_size=(3, 3), padding=1
        )
        self.to_logvar = nn.Conv2d(
            4 * hidden_dimensions, latent_dimensions, kernel_size=(3, 3), padding=1
        )

        self.decoder_input = nn.Sequential(
            nn.Conv2d(
                latent_dimensions, hidden_dimensions * 4, kernel_size=(3, 3), padding=1
            ),
            ResidualBlock(hidden_dimensions * 4),
            ResidualBlock(hidden_dimensions * 4),
        )

        self.decoder = nn.Sequential(
            # 32x32
            ResidualBlock(hidden_dimensions * 4),
            ResidualBlock(hidden_dimensions * 4),
            nn.ConvTranspose2d(
                in_channels=hidden_dimensions * 4,
                out_channels=hidden_dimensions * 2,
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dimensions * 2),
            nn.SiLU(),
            ResidualBlock(hidden_dimensions * 2),
            ResidualBlock(hidden_dimensions * 2),
            # 64x64
            nn.ConvTranspose2d(
                in_channels=hidden_dimensions * 2,
                out_channels=hidden_dimensions,
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dimensions),
            nn.SiLU(),
            ResidualBlock(hidden_dimensions),
            ResidualBlock(hidden_dimensions),
            # 128x128
            nn.ConvTranspose2d(
                in_channels=hidden_dimensions,
                out_channels=input_channels,
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.Tanh(),  # Output Data Normalized to [-1,1]
            # 256x256
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.to_mu(h)
        logvar = torch.clamp(self.to_logvar(h), min=-20, max=20)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # std == sigma, logvar == 2ln sigma --> exp((2ln sigma) / 2) == sigma
        std = torch.exp(logvar / 2)

        epsilon = torch.randn_like(std)

        # Return the new parameter, the mean + some fraction of std
        # with the fraction being sampled from a normal distribution
        return mu + (std * epsilon)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.decoder_input(z)
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
