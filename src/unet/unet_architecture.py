import torch.nn as nn
import torch

from utils.ResidualBlock import ResidualBlock


class ConvolutionBlock(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, num_groups: int = 8
    ) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=output_channels),
            nn.SiLU(),
            ResidualBlock(output_channels),
            ResidualBlock(output_channels),
        )

    def forward(self, x):
        return self.conv_block(x)


class UNet(nn.Module):
    def __init__(self, input_latent_channels: int, hidden_dimensions: int = 16) -> None:
        super().__init__()

        self.initial_conv_down = nn.Sequential(
            ConvolutionBlock(input_latent_channels, hidden_dimensions),
            ConvolutionBlock(hidden_dimensions, hidden_dimensions),
        )

        self.first_downscale = nn.Sequential(
            nn.Conv2d(
                hidden_dimensions, hidden_dimensions, kernel_size=(2, 2), stride=2
            ),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dimensions),
            nn.SiLU(),
        )

        self.second_conv_down = nn.Sequential(
            ConvolutionBlock(hidden_dimensions, hidden_dimensions * 2),
            ConvolutionBlock(hidden_dimensions * 2, hidden_dimensions * 2),
        )

        self.second_downscale = nn.Sequential(
            nn.Conv2d(
                hidden_dimensions * 2,
                hidden_dimensions * 2,
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dimensions * 2),
            nn.SiLU(),
        )

        self.third_conv_down = nn.Sequential(
            ConvolutionBlock(hidden_dimensions * 2, hidden_dimensions * 4),
            ConvolutionBlock(hidden_dimensions * 4, hidden_dimensions * 4),
        )

        self.third_downscale = nn.Sequential(
            nn.Conv2d(
                hidden_dimensions * 4,
                hidden_dimensions * 4,
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dimensions * 4),
            nn.SiLU(),
        )

        self.fourth_conv_down = nn.Sequential(
            ConvolutionBlock(hidden_dimensions * 4, hidden_dimensions * 8),
            ConvolutionBlock(hidden_dimensions * 8, hidden_dimensions * 8),
        )
        self.fourth_downscale = nn.Sequential(
            nn.Conv2d(
                hidden_dimensions * 8,
                hidden_dimensions * 8,
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dimensions * 8),
            nn.SiLU(),
        )

        self.final_conv = nn.Sequential(
            ConvolutionBlock(hidden_dimensions * 8, hidden_dimensions * 16),
            ConvolutionBlock(hidden_dimensions * 16, hidden_dimensions * 16),
        )

        self.first_upscale = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dimensions * 16,
                hidden_dimensions * 16,
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dimensions * 16),
            nn.SiLU(),
        )

        self.first_conv_up = nn.Sequential(
            ConvolutionBlock(
                hidden_dimensions * 16 + hidden_dimensions * 8, hidden_dimensions * 8
            ),
            ConvolutionBlock(hidden_dimensions * 8, hidden_dimensions * 8),
        )

        self.second_upscale = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dimensions * 8,
                hidden_dimensions * 8,
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dimensions * 8),
            nn.SiLU(),
        )

        self.second_conv_up = nn.Sequential(
            ConvolutionBlock(
                hidden_dimensions * 8 + hidden_dimensions * 4, hidden_dimensions * 4
            ),
            ConvolutionBlock(hidden_dimensions * 4, hidden_dimensions * 4),
        )

        self.third_upscale = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dimensions * 4,
                hidden_dimensions * 4,
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dimensions * 4),
            nn.SiLU(),
        )
        self.third_conv_up = nn.Sequential(
            ConvolutionBlock(
                hidden_dimensions * 4 + hidden_dimensions * 2, hidden_dimensions * 2
            ),
            ConvolutionBlock(hidden_dimensions * 2, hidden_dimensions * 2),
        )
        self.fourth_upscale = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dimensions * 2,
                hidden_dimensions * 2,
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dimensions * 2),
            nn.SiLU(),
        )
        self.output_layer = nn.Sequential(
            ConvolutionBlock(
                hidden_dimensions * 2 + hidden_dimensions, hidden_dimensions
            ),
            ConvolutionBlock(hidden_dimensions, hidden_dimensions),
            nn.Conv2d(
                hidden_dimensions, input_latent_channels, kernel_size=(3, 3), padding=1
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError("Width and Height must be divisible by 16")

        first_layer = self.initial_conv_down(x)
        x = self.first_downscale(first_layer)

        second_layer = self.second_conv_down(x)
        x = self.second_downscale(second_layer)

        third_layer = self.third_conv_down(x)
        x = self.third_downscale(third_layer)

        fourth_layer = self.fourth_conv_down(x)
        x = self.fourth_downscale(fourth_layer)

        x = self.final_conv(x)  # Bottleneck

        x = self.first_upscale(x)
        x = torch.cat([x, fourth_layer], dim=1)
        x = self.first_conv_up(x)

        x = self.second_upscale(x)
        x = torch.cat([x, third_layer], dim=1)
        x = self.second_conv_up(x)

        x = self.third_upscale(x)
        x = torch.cat([x, second_layer], dim=1)
        x = self.third_conv_up(x)

        x = self.fourth_upscale(x)
        x = torch.cat([x, first_layer], dim=1)
        x = self.output_layer(x)

        return x
