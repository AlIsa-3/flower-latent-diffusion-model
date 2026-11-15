import torch
import torch.nn as nn


class SpatialSelfAttention(nn.Module):
    def __init__(self, channels, num_heads: int = 4, num_groups: int = 8):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

        self.attention = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )
        self.projection = nn.Linear(in_features=channels, out_features=channels)

        self.attention_coef = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)
        attention, _ = self.attention(x, x, x, need_weights=False)

        attention = self.projection(attention)
        attention = attention.transpose(1, 2).reshape(B, C, H, W)

        return residual + (self.attention_coef * attention)
