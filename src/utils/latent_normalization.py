import torch
import torch.nn as nn


class LatentNormalizer(nn.Module):
    def __init__(self, scale: torch.Tensor, shift: torch.Tensor):
        super().__init__()

        self.register_buffer("scale", scale.reshape(1, -1, 1, 1))
        self.register_buffer("shift", shift.reshape(1, -1, 1, 1))

    def normalize(self, z):
        return (z - self.shift) * self.scale

    def unnormalize(self, generation):
        return (generation / self.scale) + self.shift

    def forward(self, z):
        return self.normalize(z)
