"""Autoencoder model definitions."""

from torch import Tensor, nn


class Autoencoder(nn.Module):
    """A simple fully connected autoencoder for flattened inputs."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be greater than 0")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be greater than 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than 0")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: Tensor) -> Tensor:
        # Convert batched inputs into latent vectors.
        x = x.view(x.size(0), -1)
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        # Reconstruct flattened inputs from latent vectors.
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction
