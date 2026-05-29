"""
vae.py — Challenge 6 / Group 8
Variational AutoEncoder (VAE) with β-VAE support and KL warm-up.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Variational AutoEncoder for tabular data.

    Encoder outputs (mu, logvar); decoder takes a sampled z.
    Use mu (not sampled z) for downstream visualisation — it is the
    deterministic, noise-free latent representation.

    beta > 1  →  β-VAE (more disentangled but worse reconstruction).
    """

    def __init__(
        self,
        input_dim:   int,
        hidden_dims: list[int],
        latent_dim:  int,
    ) -> None:
        super().__init__()

        # ── Encoder body ──────────────────────────────────────────────
        enc: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder_body = nn.Sequential(*enc)

        # μ and log σ² heads
        self.fc_mu     = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # ── Decoder ───────────────────────────────────────────────────
        dec: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec.append(nn.Linear(prev, input_dim))   # no activation on output
        self.decoder = nn.Sequential(*dec)

    # ── Forward pass ──────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_body(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(
        self,
        mu:     torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterisation trick: z = μ + σ·ε,  ε ~ N(0,I)."""
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (x_hat, mu, logvar)."""
        mu, logvar = self.encode(x)
        z          = self.reparameterise(mu, logvar)
        x_hat      = self.decode(z)
        return x_hat, mu, logvar

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE using μ (deterministic, no sampling noise)."""
        with torch.no_grad():
            mu, _ = self.encode(x)
            x_hat  = self.decode(mu)
        return ((x - x_hat) ** 2).mean(dim=1)


# ── Loss function ──────────────────────────────────────────────────────────

def vae_loss(
    x_hat:  torch.Tensor,
    x:      torch.Tensor,
    mu:     torch.Tensor,
    logvar: torch.Tensor,
    beta:   float = 1.0,
    warmup_factor: float = 1.0,   # 0→1 ramp during KL warm-up
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ELBO loss: reconstruction (MSE) + β·KL-divergence.
    Returns (total_loss, recon_loss, kl_loss) — all per-sample averaged.

    warmup_factor: linearly ramp the KL weight from 0 to beta over the
    first N epochs to avoid posterior collapse.
    """
    N = x.size(0)

    # Reconstruction: sum over features, average over batch
    recon = F.mse_loss(x_hat, x, reduction="sum") / N

    # KL: −0.5 Σ(1 + log σ² − μ² − σ²)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / N

    total = recon + beta * warmup_factor * kl
    return total, recon, kl
