import torch.nn as nn
import torch

from utils.UNetResidualBlock import UNetResidualBlock
from unet.diffusion_utils import SinusoidalTimestepEmbedding


class TimeStepSequential(nn.Sequential):
    # Add default to satisfy type checking
    def forward(self, input: torch.Tensor, time_embedding: torch.Tensor | None = None):
        for layer in self:
            input = layer(input, time_embedding)

        return input


class ConvolutionBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_groups: int = 8,
        timestep_embedding_dims: int = 256,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=output_channels),
            nn.SiLU(),
        )
        self.residual1 = UNetResidualBlock(output_channels, timestep_embedding_dims)
        self.residual2 = UNetResidualBlock(output_channels, timestep_embedding_dims)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor):
        x = self.conv(x)
        x = self.residual1(x, time_embedding)
        x = self.residual2(x, time_embedding)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        input_latent_channels: int,
        hidden_dimensions: int = 16,
        timestep_embedding_dims: int = 256,
    ) -> None:
        super().__init__()

        if hidden_dimensions % 8 != 0:
            raise ValueError("hidden_dimensions must be divisible by 8")

        # Allow model to learn a more useful representation of the timestep embedding
        self.timestep_embedding = nn.Sequential(
            SinusoidalTimestepEmbedding(embedding_dimensions=64),
            nn.Linear(64, timestep_embedding_dims),
            nn.SiLU(),
            nn.Linear(timestep_embedding_dims, timestep_embedding_dims),
        )

        self.initial_conv_down = TimeStepSequential(
            ConvolutionBlock(
                input_latent_channels,
                hidden_dimensions,
                timestep_embedding_dims=timestep_embedding_dims,
            ),
            ConvolutionBlock(
                hidden_dimensions,
                hidden_dimensions,
                timestep_embedding_dims=timestep_embedding_dims,
            ),
        )

        self.first_downscale = nn.Sequential(
            nn.Conv2d(
                hidden_dimensions, hidden_dimensions, kernel_size=(2, 2), stride=2
            ),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dimensions),
            nn.SiLU(),
        )

        self.second_conv_down = TimeStepSequential(
            ConvolutionBlock(
                hidden_dimensions,
                hidden_dimensions * 2,
                timestep_embedding_dims=timestep_embedding_dims,
            ),
            ConvolutionBlock(
                hidden_dimensions * 2,
                hidden_dimensions * 2,
                timestep_embedding_dims=timestep_embedding_dims,
            ),
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

        self.third_conv_down = TimeStepSequential(
            ConvolutionBlock(
                hidden_dimensions * 2,
                hidden_dimensions * 4,
                timestep_embedding_dims=timestep_embedding_dims,
            ),
            ConvolutionBlock(
                hidden_dimensions * 4,
                hidden_dimensions * 4,
                timestep_embedding_dims=timestep_embedding_dims,
            ),
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

        self.fourth_conv_down = TimeStepSequential(
            ConvolutionBlock(
                hidden_dimensions * 4,
                hidden_dimensions * 8,
                timestep_embedding_dims=timestep_embedding_dims,
            ),
            ConvolutionBlock(
                hidden_dimensions * 8,
                hidden_dimensions * 8,
                timestep_embedding_dims=timestep_embedding_dims,
            ),
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

        self.final_conv = TimeStepSequential(
            ConvolutionBlock(
                hidden_dimensions * 8,
                hidden_dimensions * 16,
                timestep_embedding_dims=timestep_embedding_dims,
            ),
            ConvolutionBlock(
                hidden_dimensions * 16,
                hidden_dimensions * 16,
                timestep_embedding_dims=timestep_embedding_dims,
            ),
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

        self.first_conv_up = TimeStepSequential(
            ConvolutionBlock(
                hidden_dimensions * 16 + hidden_dimensions * 8,
                hidden_dimensions * 8,
                timestep_embedding_dims=timestep_embedding_dims,
            ),
            ConvolutionBlock(
                hidden_dimensions * 8,
                hidden_dimensions * 8,
                timestep_embedding_dims=timestep_embedding_dims,
            ),
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

        self.second_conv_up = TimeStepSequential(
            ConvolutionBlock(
                hidden_dimensions * 8 + hidden_dimensions * 4,
                hidden_dimensions * 4,
                timestep_embedding_dims=timestep_embedding_dims,
            ),
            ConvolutionBlock(
                hidden_dimensions * 4,
                hidden_dimensions * 4,
                timestep_embedding_dims=timestep_embedding_dims,
            ),
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
        self.third_conv_up = TimeStepSequential(
            ConvolutionBlock(
                hidden_dimensions * 4 + hidden_dimensions * 2,
                hidden_dimensions * 2,
                timestep_embedding_dims=timestep_embedding_dims,
            ),
            ConvolutionBlock(
                hidden_dimensions * 2,
                hidden_dimensions * 2,
                timestep_embedding_dims=timestep_embedding_dims,
            ),
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
        self.output_block = TimeStepSequential(
            ConvolutionBlock(
                hidden_dimensions * 2 + hidden_dimensions,
                hidden_dimensions,
                timestep_embedding_dims=timestep_embedding_dims,
            ),
            ConvolutionBlock(
                hidden_dimensions,
                hidden_dimensions,
                timestep_embedding_dims=timestep_embedding_dims,
            ),
        )
        self.output_conv = nn.Conv2d(
            hidden_dimensions, input_latent_channels, kernel_size=(3, 3), padding=1
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError("Width and Height must be divisible by 16")

        time_embedding = self.timestep_embedding(timestep)

        first_layer = self.initial_conv_down(x, time_embedding)
        x = self.first_downscale(first_layer)

        second_layer = self.second_conv_down(x, time_embedding)
        x = self.second_downscale(second_layer)

        third_layer = self.third_conv_down(x, time_embedding)
        x = self.third_downscale(third_layer)

        fourth_layer = self.fourth_conv_down(x, time_embedding)
        x = self.fourth_downscale(fourth_layer)

        x = self.final_conv(x, time_embedding)  # Bottleneck

        x = self.first_upscale(x)
        x = torch.cat([x, fourth_layer], dim=1)
        x = self.first_conv_up(x, time_embedding)

        x = self.second_upscale(x)
        x = torch.cat([x, third_layer], dim=1)
        x = self.second_conv_up(x, time_embedding)

        x = self.third_upscale(x)
        x = torch.cat([x, second_layer], dim=1)
        x = self.third_conv_up(x, time_embedding)

        x = self.fourth_upscale(x)
        x = torch.cat([x, first_layer], dim=1)
        x = self.output_block(x, time_embedding)
        x = self.output_conv(x)

        return x
