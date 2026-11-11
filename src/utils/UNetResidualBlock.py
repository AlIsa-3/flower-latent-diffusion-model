import torch.nn as nn
import torch
from utils.ResidualBlock import ResidualBlock


class UNetResidualBlock(ResidualBlock):
    def __init__(self, channels: int, time_embedding_dimensions: int = 256):
        super().__init__(channels)
        # Project time embedding to the correct number of channels
        self.adagn = AdaGN(
            channels=channels, embedding_dims=time_embedding_dimensions, num_groups=8
        )

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    # Add default time_embedding to avoid typing warning
    def forward(
        self, x: torch.Tensor, time_embedding: torch.Tensor | None = None
    ) -> torch.Tensor:
        if time_embedding is None:
            raise ValueError("time_embedding is required")
        residual = x
        x = self.conv1(x)
        x = self.groupNorm1(x)
        x = self.activation(x)

        # Add time embedding
        x = self.adagn(x, time_embedding)

        x = self.conv2(x)
        x = self.groupNorm2(x)
        x = x + residual
        x = self.activation(x)

        return x


class AdaGN(nn.Module):
    def __init__(self, channels, embedding_dims: int, num_groups: int = 8):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.projection = nn.Linear(embedding_dims, channels * 2)

        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x, time_embedding):
        scale, shift = self.projection(time_embedding).chunk(2, dim=1)

        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        x = self.norm(x)
        return x * (1 + scale) + shift
