"""
autoencoder.py — Challenge 6 / Group 8
Standard AutoEncoder (AE) for anomaly detection on tabular data.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """
    Symmetric bottleneck AE:
      input → hidden_dims → latent_dim → hidden_dims (reversed) → input

    Example: input=12, hidden_dims=[128,64], latent_dim=16
      Encoder: 12→128→64→16
      Decoder: 16→64→128→12
    """

    def __init__(
        self,
        input_dim:   int,
        hidden_dims: list[int],
        latent_dim:  int,
    ) -> None:
        super().__init__()

        # ── Encoder ───────────────────────────────────────────────────
        enc_layers: list[nn.Module] = []
        dims = [input_dim] + hidden_dims + [latent_dim]
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:          # no ReLU on the latent layer
                enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder ───────────────────────────────────────────────────
        dec_layers: list[nn.Module] = []
        rdims = [latent_dim] + list(reversed(hidden_dims)) + [input_dim]
        for i in range(len(rdims) - 1):
            dec_layers.append(nn.Linear(rdims[i], rdims[i + 1]))
            if i < len(rdims) - 2:          # no activation on output
                dec_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z    = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE (no grad required at inference)."""
        with torch.no_grad():
            x_hat, _ = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=1)
