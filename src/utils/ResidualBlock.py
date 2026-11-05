import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), padding=1)
        self.groupNorm1 = nn.GroupNorm(8, channels)
        self.groupNorm2 = nn.GroupNorm(8, channels)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.groupNorm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.groupNorm2(x)
        x = x + residual
        x = self.activation(x)

        return x
