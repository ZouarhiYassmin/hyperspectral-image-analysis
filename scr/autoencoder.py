"""
Convolutional Autoencoder for Hyperspectral Image Compression.
Treats spectral bands as channels (C x H x W).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class AEConfig:
    in_channels: int = 200      # number of spectral bands
    latent_dim: int = 32
    hidden_channels: list = None
    lambda_sam: float = 0.1     # weight of spectral loss

    def __post_init__(self):
        if self.hidden_channels is None:
            self.hidden_channels = [64, 32]


class Encoder(nn.Module):
    def __init__(self, cfg: AEConfig, spatial_size: int = 64):
        super().__init__()
        channels = [cfg.in_channels] + cfg.hidden_channels

        layers = []
        for i in range(len(channels) - 1):
            layers += [
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.conv = nn.Sequential(*layers)

        # Compute flattened size after conv
        n_downsamples = len(cfg.hidden_channels)
        reduced = spatial_size // (2 ** n_downsamples)
        flat_dim = cfg.hidden_channels[-1] * reduced * reduced
        self.fc = nn.Linear(flat_dim, cfg.latent_dim)
        self._flat_dim = flat_dim
        self._reduced = reduced

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, cfg: AEConfig, spatial_size: int = 64):
        super().__init__()
        n_downsamples = len(cfg.hidden_channels)
        self._reduced = spatial_size // (2 ** n_downsamples)
        self._start_ch = cfg.hidden_channels[-1]

        flat_dim = self._start_ch * self._reduced * self._reduced
        self.fc = nn.Linear(cfg.latent_dim, flat_dim)

        channels = list(reversed(cfg.hidden_channels)) + [cfg.in_channels]
        layers = []
        for i in range(len(channels) - 1):
            is_last = (i == len(channels) - 2)
            layers += [
                nn.ConvTranspose2d(
                    channels[i], channels[i + 1],
                    kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.Sigmoid() if is_last else nn.BatchNorm2d(channels[i + 1]),
            ]
            if not is_last:
                layers.append(nn.ReLU(inplace=True))
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self._start_ch, self._reduced, self._reduced)
        return self.deconv(x)


class ConvAutoencoder(nn.Module):
    def __init__(self, cfg: AEConfig, spatial_size: int = 64):
        super().__init__()
        self.encoder = Encoder(cfg, spatial_size)
        self.decoder = Decoder(cfg, spatial_size)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def compress(self, x):
        """Return latent codes."""
        with torch.no_grad():
            return self.encoder(x)

    def decompress(self, z):
        """Reconstruct from latent codes."""
        with torch.no_grad():
            return self.decoder(z)