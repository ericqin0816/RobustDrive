import torch
from torch import nn


class NvidiaCNN(nn.Module):
    """Five convolutions and a compact regressor, inspired by Bojarski et al. (2016)."""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        layers = []
        for inputs, outputs, kernel, stride in (
            (3, 24, 5, 2),
            (24, 36, 5, 2),
            (36, 48, 5, 2),
            (48, 64, 3, 1),
            (64, 64, 3, 1),
        ):
            layers.extend([nn.Conv2d(inputs, outputs, kernel, stride), nn.ReLU()])
        self.features = nn.Sequential(*layers)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1152, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(images)).squeeze(-1)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
