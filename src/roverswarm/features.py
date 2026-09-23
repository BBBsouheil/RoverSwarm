from __future__ import annotations

import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SpatialFeatureExtractor(BaseFeaturesExtractor):
    """Encode la carte partiellement connue sans perdre sa structure spatiale."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        grid_shape: tuple[int, int],
        channels: int = 5,
        scalar_size: int = 9,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)
        self.grid_shape = tuple(grid_shape)
        self.channels = channels
        self.scalar_size = scalar_size
        self.spatial_size = channels * self.grid_shape[0] * self.grid_shape[1]
        expected = self.spatial_size + scalar_size
        if observation_space.shape != (expected,):
            raise ValueError(f"Observation spatiale attendue de taille {expected}, reçue {observation_space.shape}")
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, channels, *self.grid_shape)
            cnn_size = int(self.cnn(sample).shape[1])
        self.scalar_net = nn.Sequential(nn.Linear(scalar_size, 32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(cnn_size + 32, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        spatial = observations[:, :self.spatial_size].reshape(
            -1, self.channels, self.grid_shape[0], self.grid_shape[1]
        )
        scalars = observations[:, self.spatial_size:self.spatial_size + self.scalar_size]
        return self.fusion(torch.cat((self.cnn(spatial), self.scalar_net(scalars)), dim=1))
