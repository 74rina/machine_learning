"""MLP baseline for dissolved inorganic carbon (DIC) regression."""

from torch import nn


class MLPRegressor(nn.Module):
    """Map numeric features through 64 and 32 ReLU units to one value."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features):
        return self.layers(features)
