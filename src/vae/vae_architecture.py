import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_channels, hidden_dimensions, latent_dimensions=4):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=hidden_dimensions,
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.BatchNorm2d(hidden_dimensions),
            nn.ReLU(),
            # 128x128
            nn.Conv2d(
                in_channels=hidden_dimensions,
                out_channels=hidden_dimensions * 2,
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.BatchNorm2d(hidden_dimensions * 2),
            nn.ReLU(),
            # 64x64
            nn.Conv2d(
                in_channels=hidden_dimensions * 2,
                out_channels=hidden_dimensions * 4,
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.BatchNorm2d(hidden_dimensions * 4),
            nn.ReLU(),
            # 32x32
            nn.Conv2d(
                in_channels=hidden_dimensions * 4,
                out_channels=hidden_dimensions * 8,
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.BatchNorm2d(hidden_dimensions * 8),
            nn.ReLU(),
            # 16x16
        )
        # Convert Encoded Data to mu and logvar
        self.to_mu = nn.Conv2d(
            8 * hidden_dimensions, latent_dimensions, kernel_size=(3, 3), padding=1
        )
        self.to_logvar = nn.Conv2d(
            8 * hidden_dimensions, latent_dimensions, kernel_size=(3, 3), padding=1
        )

        self.decoder_input = nn.Conv2d(
            latent_dimensions, hidden_dimensions * 8, kernel_size=(3, 3), padding=1
        )

        self.decoder = nn.Sequential(
            # 16x16
            nn.ConvTranspose2d(
                in_channels=hidden_dimensions * 8,
                out_channels=hidden_dimensions * 4,
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.BatchNorm2d(hidden_dimensions * 4),
            nn.ReLU(),
            # 32x32
            nn.ConvTranspose2d(
                in_channels=hidden_dimensions * 4,
                out_channels=hidden_dimensions * 2,
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.BatchNorm2d(hidden_dimensions * 2),
            nn.ReLU(),
            # 64x64
            nn.ConvTranspose2d(
                in_channels=hidden_dimensions * 2,
                out_channels=hidden_dimensions,
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.BatchNorm2d(hidden_dimensions),
            nn.ReLU(),
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

    def encode(self, x):
        h = self.encoder(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # std == sigma, logvar == 2ln sigma --> exp((2ln sigma) / 2) == sigma
        std = torch.exp(logvar / 2)

        epsilon = torch.randn_like(std)

        # Return the new parameter, the mean + some fraction of std
        # with the fraction being sampled from a normal distribution
        return mu + (std * epsilon)

    def decode(self, z):
        z = self.decoder_input(z)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
