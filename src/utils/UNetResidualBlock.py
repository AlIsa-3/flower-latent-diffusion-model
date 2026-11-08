import torch.nn as nn
import torch
from utils.ResidualBlock import ResidualBlock


class UNetResidualBlock(ResidualBlock):
    def __init__(self, channels: int, time_embedding_dimensions: int = 256):
        super().__init__(channels)
        # Project time embedding to the correct number of channels
        self.time_embedding = nn.Linear(time_embedding_dimensions, channels)

    # Add default time_embedding to avoid typing warning
    def forward(
        self, x: torch.Tensor, time_embedding: torch.Tensor = torch.ones(1)
    ) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.groupNorm1(x)
        x = self.activation(x)

        # Add time embedding
        time_projection = self.time_embedding(time_embedding)
        time_projection = time_projection.unsqueeze(-1).unsqueeze(
            -1
        )  # Add empty spatial dimensions (B,C,1,1)
        x = x + time_projection

        x = self.conv2(x)
        x = self.groupNorm2(x)
        x = x + residual
        x = self.activation(x)

        return x
