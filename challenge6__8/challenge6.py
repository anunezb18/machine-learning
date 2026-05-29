"""
challenge6.py — Challenge 6: AutoEncoders & Representation Learning
====================================================================
Group 8 — Energy & Utilities
Dataset: EIA Electricity Retail Sales (same preprocessing as Challenge 5)

Three models:
  1. AutoEncoder (AE)         — anomaly detection via reconstruction error
  2. Variational AutoEncoder  — probabilistic latent space + β sweep
  3. Isolation Forest         — classical baseline

Visualisation: t-SNE and UMAP on raw features, AE latent space,
               VAE μ vectors; coloured by anomaly score and Ch5 clusters.

Run:
    python challenge6.py

All figures → figures/c6_*
All metrics → results/c6_*
Model weights → models/
"""

from __future__ import annotations

import os
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*n_jobs.*", category=UserWarning)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

# umap-learn is installed separately: pip install umap-learn
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("WARNING: umap-learn not installed. "
          "Run: pip install umap-learn")

from autoencoder import AutoEncoder
from vae import VAE, vae_loss

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────
CACHE_FILE = "eia_retail_sales.csv"
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("models",  exist_ok=True)
os.makedirs("logs",    exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/challenge6.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  PREPROCESSING  (identical to Challenge 5)
# ═══════════════════════════════════════════════════════════════════════════

def load_and_preprocess():
    """
    Same pipeline as Challenge 5.
    Returns:
        X_scaled  : (N, 12)  StandardScaler-normalised features
        df        : original DataFrame with period/stateid columns
        FEATURES  : list of feature names
        ch5_labels: K-Means k=2 labels from Challenge 5 (for visualisation)
    """
    log.info(f"Loading data from {CACHE_FILE}")
    df_raw = pd.read_csv(CACHE_FILE)

    df = df_raw[df_raw["sectorid"] == "ALL"].copy()
    for col in ["sales", "price", "revenue", "customers"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["period"] = pd.to_datetime(df["period"])
    df = df.sort_values(["stateid", "period"]).reset_index(drop=True)
    df = df.dropna(subset=["sales"]).reset_index(drop=True)

    grp = df.groupby("stateid")
    df["lag1"]        = grp["sales"].shift(1)
    df["lag12"]       = grp["sales"].shift(12)
    _past = grp["sales"].transform(lambda x: x.shift(1))
    df["roll12_mean"] = _past.rolling(12, min_periods=6).mean()
    df["roll12_std"]  = _past.rolling(12, min_periods=6).std()
    df["yoy_growth"]  = (
        (df["sales"] - df["lag12"]) /
        (df["lag12"].replace(0, np.nan) + 1e-9)
    ).clip(-5, 5)
    df["revenue_per_mwh"] = (
        df["revenue"] / (df["sales"].replace(0, np.nan) + 1e-9)
    )
    df["month_sin"]  = np.sin(2 * np.pi * df["period"].dt.month / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["period"].dt.month / 12)
    df["year_trend"] = (df["period"].dt.year - 2001)*12 + df["period"].dt.month
    df["state_enc"]  = df["stateid"].astype("category").cat.codes

    FEATURES = ["sales","price","revenue_per_mwh","lag1","lag12",
                "roll12_mean","roll12_std","yoy_growth",
                "month_sin","month_cos","year_trend","state_enc"]

    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    X_raw    = df[FEATURES].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    log.info(f"  Dataset: {X_scaled.shape[0]:,} rows × {X_scaled.shape[1]} features")

    # Reproduce Challenge 5 K-Means k=2 labels for cross-challenge coloring
    km = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=SEED)
    ch5_labels = km.fit_predict(X_scaled)
    log.info(f"  Ch5 cluster sizes: {np.bincount(ch5_labels)}")

    return X_scaled, df, FEATURES, ch5_labels


# ═══════════════════════════════════════════════════════════════════════════
# 2.  TRAIN / TEST SPLIT  (20% held-out, stratified on Ch5 clusters)
# ═══════════════════════════════════════════════════════════════════════════

def train_test_split_indices(N: int, ch5_labels: np.ndarray, test_frac=0.20):
    """Stratified 80/20 split. Returns train_idx, test_idx."""
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac,
                                 random_state=SEED)
    idx = np.arange(N)
    train_idx, test_idx = next(sss.split(idx, ch5_labels))
    log.info(f"  Train: {len(train_idx):,}  |  Test: {len(test_idx):,}")
    return train_idx, test_idx


# ═══════════════════════════════════════════════════════════════════════════
# 3.  AUTOENCODER TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_autoencoder(
    X_train: np.ndarray,
    input_dim: int,
    hidden_dims: list[int],
    latent_dim: int,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = SEED,
    tag: str = "ae",
) -> tuple[AutoEncoder, list[float]]:
    """Train AE and return (model, loss_history)."""
    torch.manual_seed(seed)
    model = AutoEncoder(input_dim, hidden_dims, latent_dim).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.MSELoss()

    X_t  = torch.tensor(X_train, dtype=torch.float32)
    ds   = TensorDataset(X_t)
    dl   = DataLoader(ds, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            x_hat, _ = model(xb)
            loss = crit(x_hat, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(X_train)
        losses.append(ep_loss)
        if epoch % 20 == 0:
            log.info(f"  AE [{tag}] epoch {epoch:>3}/{epochs}  "
                     f"loss={ep_loss:.6f}")

    torch.save(model.state_dict(), f"models/{tag}.pt")
    return model, losses


# ═══════════════════════════════════════════════════════════════════════════
# 4.  VAE TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_vae(
    X_train: np.ndarray,
    input_dim: int,
    hidden_dims: list[int],
    latent_dim: int,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    beta: float = 1.0,
    warmup_epochs: int = 20,
    seed: int = SEED,
    tag: str = "vae",
) -> tuple[VAE, list[float], list[float], list[float]]:
    """
    Train VAE with KL warm-up (0→beta over warmup_epochs).
    Returns (model, total_losses, recon_losses, kl_losses).
    """
    torch.manual_seed(seed)
    model = VAE(input_dim, hidden_dims, latent_dim).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    ds  = TensorDataset(X_t)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True)

    total_hist, recon_hist, kl_hist = [], [], []

    for epoch in range(1, epochs + 1):
        # Linear KL warm-up: 0 → beta over warmup_epochs
        warmup = min(1.0, epoch / max(1, warmup_epochs))

        model.train()
        ep_total = ep_recon = ep_kl = 0.0
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            x_hat, mu, logvar = model(xb)
            loss, recon, kl = vae_loss(x_hat, xb, mu, logvar,
                                       beta=beta, warmup_factor=warmup)
            opt.zero_grad(); loss.backward(); opt.step()
            n = len(xb)
            ep_total += loss.item() * n
            ep_recon += recon.item() * n
            ep_kl    += kl.item()   * n

        N = len(X_train)
        total_hist.append(ep_total / N)
        recon_hist.append(ep_recon / N)
        kl_hist.append(ep_kl / N)

        if epoch % 20 == 0:
            log.info(f"  VAE [{tag}] epoch {epoch:>3}/{epochs}  "
                     f"total={total_hist[-1]:.4f}  "
                     f"recon={recon_hist[-1]:.4f}  "
                     f"kl={kl_hist[-1]:.4f}  "
                     f"warmup={warmup:.2f}")

    torch.save(model.state_dict(), f"models/{tag}.pt")
    return model, total_hist, recon_hist, kl_hist


# ═══════════════════════════════════════════════════════════════════════════
# 5.  ANOMALY SCORING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def get_ae_errors(model: AutoEncoder, X: np.ndarray) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    return model.reconstruction_error(X_t).cpu().numpy()


def get_vae_errors(model: VAE, X: np.ndarray) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    return model.reconstruction_error(X_t).cpu().numpy()


def get_vae_latent(model: VAE, X: np.ndarray) -> np.ndarray:
    """Extract μ vectors (deterministic latent representation)."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        mu, _ = model.encode(X_t)
    return mu.cpu().numpy()


def get_ae_latent(model: AutoEncoder, X: np.ndarray) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        z = model.encode(X_t)
    return z.cpu().numpy()


def anomaly_threshold(errors: np.ndarray,
                      method: str = "p95") -> tuple[float, np.ndarray]:
    """
    Compute threshold and return boolean anomaly mask.
    method: 'p95' | 'p99' | 'mean3sigma'
    """
    if method == "p95":
        thr = float(np.percentile(errors, 95))
    elif method == "p99":
        thr = float(np.percentile(errors, 99))
    else:  # mean + 3σ
        thr = float(errors.mean() + 3 * errors.std())
    return thr, errors > thr


# ═══════════════════════════════════════════════════════════════════════════
# 6.  FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def plot_loss_curves(ae_losses, vae_total, vae_recon, vae_kl):
    """Figure (a)+(b): AE training curve and VAE component curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(ae_losses, lw=1.5, color="steelblue")
    ax1.set(xlabel="Epoch", ylabel="MSE Loss",
            title="AE — Training Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(vae_total, lw=1.5, label="Total",        color="black")
    ax2.plot(vae_recon, lw=1.2, label="Reconstruction",color="steelblue", ls="--")
    ax2.plot(vae_kl,    lw=1.2, label="KL",            color="tomato",    ls="--")
    ax2.set(xlabel="Epoch", ylabel="Loss",
            title="VAE — Training Loss (Reconstruction + KL)")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("figures/c6_01_loss_curves.png", dpi=120)
    plt.close(fig)
    log.info("  Saved: figures/c6_01_loss_curves.png")


def plot_error_histograms(ae_errors, vae_errors, thr_ae, thr_vae):
    """Figure (c): reconstruction error distributions with thresholds."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.hist(ae_errors, bins=80, color="steelblue", alpha=0.75, edgecolor="none")
    ax1.axvline(thr_ae, color="red", lw=2, label=f"p95 threshold={thr_ae:.4f}")
    ax1.set(xlabel="Reconstruction Error (MSE)", ylabel="Count",
            title="AE — Error Distribution")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.hist(vae_errors, bins=80, color="salmon", alpha=0.75, edgecolor="none")
    ax2.axvline(thr_vae, color="darkred", lw=2,
                label=f"p95 threshold={thr_vae:.4f}")
    ax2.set(xlabel="Reconstruction Error (MSE)", ylabel="Count",
            title="VAE — Error Distribution")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("figures/c6_02_error_histograms.png", dpi=120)
    plt.close(fig)
    log.info("  Saved: figures/c6_02_error_histograms.png")


def plot_2d_embeddings(Z_2d: np.ndarray,
                       color_arr: np.ndarray,
                       title: str,
                       fname: str,
                       cmap: str = "plasma",
                       legend_labels: list | None = None):
    """Generic 2-D scatter with continuous or discrete coloring."""
    fig, ax = plt.subplots(figsize=(8, 6))

    if legend_labels is not None:
        # Discrete coloring (e.g. cluster labels)
        unique = sorted(set(color_arr.astype(int)))
        cmap_d = cm.get_cmap("tab10", len(unique))
        for i, lbl in enumerate(unique):
            mask = color_arr.astype(int) == lbl
            name = legend_labels[lbl] if lbl < len(legend_labels) else f"Cluster {lbl}"
            ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
                       c=[cmap_d(i)], s=4, alpha=0.4,
                       label=name, rasterized=True)
        ax.legend(markerscale=4, fontsize=8)
    else:
        sc = ax.scatter(Z_2d[:, 0], Z_2d[:, 1],
                        c=color_arr, cmap=cmap, s=4, alpha=0.5,
                        rasterized=True)
        plt.colorbar(sc, ax=ax, label="Score")

    ax.set(title=title, xlabel="Dim 1", ylabel="Dim 2")
    fig.tight_layout()
    fig.savefig(f"figures/{fname}", dpi=120)
    plt.close(fig)
    log.info(f"  Saved: figures/{fname}")


def plot_ae_vs_iso(ae_errors, iso_scores, rho):
    """Figure (f): AE error vs Isolation Forest score scatter."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ae_errors, iso_scores, s=3, alpha=0.3, color="steelblue",
               rasterized=True)
    ax.set(xlabel="AE Reconstruction Error",
           ylabel="Isolation Forest Score",
           title=f"AE Error vs Isolation Forest  (Spearman ρ = {rho:.3f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/c6_06_ae_vs_iso.png", dpi=120)
    plt.close(fig)
    log.info("  Saved: figures/c6_06_ae_vs_iso.png")


# ═══════════════════════════════════════════════════════════════════════════
# 7.  DIMENSIONALITY REDUCTION FOR VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════

def reduce_tsne(Z: np.ndarray, perplexity: float = 30) -> np.ndarray:
    """t-SNE on the provided array (caller is responsible for subsampling)."""
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=SEED, method="barnes_hut", n_jobs=-1)
    return tsne.fit_transform(Z)


def reduce_umap(Z: np.ndarray,
                n_neighbors: int = 15,
                min_dist: float = 0.1) -> np.ndarray:
    if not UMAP_AVAILABLE:
        log.warning("  UMAP skipped (umap-learn not installed)")
        return None
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=SEED, n_jobs=1)
    return reducer.fit_transform(Z)


# ═══════════════════════════════════════════════════════════════════════════
# 8.  MULTI-SEED EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

def run_multiseed_ae(X_train, X_all, input_dim,
                     hidden_dims, latent_dim, epochs):
    """Train AE with seeds [42,123,777]; return mean±std of p95 threshold."""
    seed_errors = []
    for s in [42, 123, 777]:
        m, _ = train_autoencoder(X_train, input_dim, hidden_dims, latent_dim,
                                 epochs=epochs, seed=s, tag=f"ae_s{s}")
        errs = get_ae_errors(m, X_all)
        seed_errors.append(np.percentile(errs, 95))
    log.info(f"  AE p95 threshold across seeds: "
             f"{np.mean(seed_errors):.4f} ± {np.std(seed_errors):.4f}")
    return seed_errors


def run_multiseed_vae(X_train, X_all, input_dim,
                      hidden_dims, latent_dim, beta, epochs):
    """Train VAE with seeds [42,123,777]; report mean±std of p95 threshold."""
    seed_errors = []
    for s in [42, 123, 777]:
        m, _, _, _ = train_vae(
            X_train, input_dim, hidden_dims, latent_dim,
            epochs=epochs, beta=beta, warmup_epochs=20,
            seed=s, tag=f"vae_s{s}",
        )
        errs = get_vae_errors(m, X_all)
        seed_errors.append(np.percentile(errs, 95))
    log.info(f"  VAE p95 threshold across seeds: "
             f"{np.mean(seed_errors):.4f} ± {np.std(seed_errors):.4f}")
    return seed_errors


# ═══════════════════════════════════════════════════════════════════════════
# 9.  ANOMALY CASE STUDY
# ═══════════════════════════════════════════════════════════════════════════

def top_anomaly_report(df: pd.DataFrame,
                       ae_errors: np.ndarray,
                       iso_scores: np.ndarray,
                       n: int = 10) -> pd.DataFrame:
    """Print and save the top-n anomalies by AE error."""
    top_idx = np.argsort(ae_errors)[::-1][:n]
    report  = df.iloc[top_idx][["period","stateid","stateDescription",
                                 "sales","price"]].copy()
    report["ae_error"]  = ae_errors[top_idx].round(4)
    report["iso_score"] = iso_scores[top_idx].round(4)
    report = report.reset_index(drop=True)
    report.to_csv("results/c6_top_anomalies.csv", index=False)
    log.info(f"\nTop {n} anomalies by AE error:\n{report.to_string()}")
    return report


# ═══════════════════════════════════════════════════════════════════════════
# 10.  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    log.info("Challenge 6 — AutoEncoders & Representation Learning")
    log.info("="*60)

    # ── 1. Load data ───────────────────────────────────────────────────
    X_scaled, df, FEATURES, ch5_labels = load_and_preprocess()
    N, D = X_scaled.shape

    train_idx, test_idx = train_test_split_indices(N, ch5_labels)
    X_train = X_scaled[train_idx]
    # Anomaly scoring is computed on the FULL dataset (train + test)

    # ── 2. Isolation Forest (classical baseline) ───────────────────────
    log.info("\n── Isolation Forest ─────────────────────────────────────")
    iso = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=SEED, n_jobs=-1)
    iso.fit(X_train)
    iso_scores = -iso.score_samples(X_scaled)   # higher = more anomalous
    # Use the model's internal decision threshold (derived from contamination=0.05)
    # instead of re-applying p95 on top, which would double-count anomalies.
    thr_iso    = float(-iso.offset_)
    anomalies_iso = iso_scores > thr_iso
    log.info(f"  Isolation Forest anomaly rate: "
             f"{anomalies_iso.mean():.1%} "
             f"(threshold={thr_iso:.4f})")

    # ── 3. AutoEncoder — architecture ablation ─────────────────────────
    log.info("\n── AutoEncoder (architecture ablation) ──────────────────")
    ae_results = []
    for latent_dim in [8, 16, 32]:
        tag = f"ae_latent{latent_dim}"
        log.info(f"  Training AE latent_dim={latent_dim}")
        model_ae, ae_losses = train_autoencoder(
            X_train, input_dim=D,
            hidden_dims=[128, 64], latent_dim=latent_dim,
            epochs=100, tag=tag,
        )
        errs = get_ae_errors(model_ae, X_scaled)
        thr, mask = anomaly_threshold(errs, "p95")
        # Silhouette must be computed only on the training portion to avoid
        # implicitly selecting latent_dim based on held-out test data.
        Z_ae_train = get_ae_latent(model_ae, X_train)
        sil_latent = silhouette_score(
            Z_ae_train,
            ch5_labels[train_idx],
            sample_size=min(8000, len(train_idx)), random_state=SEED,
        )
        ae_results.append({
            "latent_dim": latent_dim,
            "mean_error": round(float(errs.mean()), 5),
            "p95_thr":    round(thr, 5),
            "anomaly_rate": round(float(mask.mean()), 4),
            "sil_latent": round(sil_latent, 4),
        })
        log.info(f"    latent={latent_dim}  p95={thr:.4f}  "
                 f"anomaly_rate={mask.mean():.1%}  "
                 f"sil_latent={sil_latent:.4f}")

    # Keep best AE (highest silhouette on latent space)
    best_ae_cfg = max(ae_results, key=lambda r: r["sil_latent"])
    best_latent_dim = best_ae_cfg["latent_dim"]
    log.info(f"  → Best AE: latent_dim={best_latent_dim}")

    model_ae, ae_losses = train_autoencoder(
        X_train, input_dim=D,
        hidden_dims=[128, 64], latent_dim=best_latent_dim,
        epochs=100, tag="ae_best",
    )
    ae_errors = get_ae_errors(model_ae, X_scaled)
    thr_ae, anomalies_ae = anomaly_threshold(ae_errors, "p95")
    Z_ae = get_ae_latent(model_ae, X_scaled)

    pd.DataFrame(ae_results).to_csv("results/c6_ae_ablation.csv", index=False)

    # Multi-seed stability
    seed_thrs = run_multiseed_ae(X_train, X_scaled, D,
                                  [128, 64], best_latent_dim, epochs=100)

    # ── 4. VAE — β sweep ──────────────────────────────────────────────
    log.info("\n── VAE (β sweep) ─────────────────────────────────────────")
    vae_results = []
    vae_losses_best = None  # for loss curve figure

    for beta in [0.5, 1.0, 4.0]:
        tag = f"vae_beta{beta}"
        log.info(f"  Training VAE β={beta}")
        model_vae, vt, vr, vk = train_vae(
            X_train, input_dim=D,
            hidden_dims=[128, 64], latent_dim=best_latent_dim,
            epochs=100, beta=beta, warmup_epochs=20,
            seed=SEED, tag=tag,
        )
        errs = get_vae_errors(model_vae, X_scaled)
        Z_mu = get_vae_latent(model_vae, X_scaled)
        thr, mask = anomaly_threshold(errs, "p95")
        # Same leakage fix: evaluate silhouette on training portion only.
        Z_mu_train = get_vae_latent(model_vae, X_train)
        sil_latent = silhouette_score(
            Z_mu_train,
            ch5_labels[train_idx],
            sample_size=min(8000, len(train_idx)), random_state=SEED,
        )
        vae_results.append({
            "beta": beta,
            "mean_error":   round(float(errs.mean()), 5),
            "p95_thr":      round(thr, 5),
            "anomaly_rate": round(float(mask.mean()), 4),
            "sil_latent":   round(sil_latent, 4),
            "final_kl":     round(vk[-1], 4),
        })
        log.info(f"    β={beta}  p95={thr:.4f}  "
                 f"sil={sil_latent:.4f}  KL_final={vk[-1]:.4f}")

        if beta == 1.0:
            vae_losses_best = (vt, vr, vk)

    best_vae_cfg = max(vae_results, key=lambda r: r["sil_latent"])
    best_beta    = best_vae_cfg["beta"]
    log.info(f"  → Best VAE: β={best_beta}")

    # Re-train best VAE for downstream use
    model_vae_best, vt_b, vr_b, vk_b = train_vae(
        X_train, input_dim=D,
        hidden_dims=[128, 64], latent_dim=best_latent_dim,
        epochs=100, beta=best_beta, warmup_epochs=20,
        seed=SEED, tag="vae_best",
    )
    vae_errors = get_vae_errors(model_vae_best, X_scaled)
    thr_vae, anomalies_vae = anomaly_threshold(vae_errors, "p95")
    Z_vae = get_vae_latent(model_vae_best, X_scaled)

    pd.DataFrame(vae_results).to_csv("results/c6_vae_beta_sweep.csv", index=False)

    # Multi-seed stability — VAE (required by challenge protocol)
    log.info("\n── VAE multi-seed stability ──────────────────────────────")
    run_multiseed_vae(X_train, X_scaled, D,
                      [128, 64], best_latent_dim, best_beta, epochs=100)

    # ── 5. Spearman correlations ───────────────────────────────────────
    log.info("\n── Detector agreement (Spearman ρ) ──────────────────────")
    rho_ae_iso,  _ = spearmanr(ae_errors,  iso_scores)
    rho_vae_iso, _ = spearmanr(vae_errors, iso_scores)
    rho_ae_vae,  _ = spearmanr(ae_errors,  vae_errors)
    log.info(f"  AE  vs ISO: ρ={rho_ae_iso:.3f}")
    log.info(f"  VAE vs ISO: ρ={rho_vae_iso:.3f}")
    log.info(f"  AE  vs VAE: ρ={rho_ae_vae:.3f}")

    # ── 6. Latent space Silhouette comparison ──────────────────────────
    # Silhouette on raw features (reference)
    pca2 = PCA(n_components=min(10, D), random_state=SEED)
    X_pca = pca2.fit_transform(X_scaled)
    sil_raw    = silhouette_score(X_pca, ch5_labels,
                                  sample_size=min(8000,N), random_state=SEED)
    sil_ae_lat = silhouette_score(Z_ae, ch5_labels,
                                  sample_size=min(8000,N), random_state=SEED)
    sil_vae_lat = silhouette_score(Z_vae, ch5_labels,
                                   sample_size=min(8000,N), random_state=SEED)
    log.info(f"\n  Silhouette (Ch5 labels):")
    log.info(f"    Raw features  : {sil_raw:.4f}")
    log.info(f"    AE latent     : {sil_ae_lat:.4f}")
    log.info(f"    VAE μ         : {sil_vae_lat:.4f}")

    # ── 7. Figures ─────────────────────────────────────────────────────
    log.info("\n── Generating figures ────────────────────────────────────")

    # (a)+(b) Loss curves
    vt_plot, vr_plot, vk_plot = (vae_losses_best if vae_losses_best
                                  else (vt_b, vr_b, vk_b))
    plot_loss_curves(ae_losses, vt_plot, vr_plot, vk_plot)

    # (c) Error histograms
    plot_error_histograms(ae_errors, vae_errors, thr_ae, thr_vae)

    # (d) t-SNE of VAE latent space coloured by anomaly score
    log.info("  Running t-SNE on VAE latent space...")
    # Use a consistent subsample index for all t-SNE / UMAP plots
    tsne_n = min(8000, N)
    tsne_idx = np.random.RandomState(SEED).choice(N, tsne_n, replace=False)

    Z_vae_tsne = reduce_tsne(Z_vae[tsne_idx], perplexity=30)
    plot_2d_embeddings(
        Z_vae_tsne, vae_errors[tsne_idx],
        "t-SNE of VAE Latent Space (coloured by anomaly score)",
        "c6_03_tsne_vae_anomaly.png", cmap="plasma",
    )

    # t-SNE coloured by Ch5 cluster labels
    plot_2d_embeddings(
        Z_vae_tsne, ch5_labels[tsne_idx],
        "t-SNE of VAE Latent Space (Ch5 cluster labels)",
        "c6_04_tsne_vae_clusters.png",
        legend_labels=["Cluster 0", "Cluster 1"],
    )

    # (e) UMAP of AE latent space coloured by Ch5 labels
    if UMAP_AVAILABLE:
        log.info("  Running UMAP on AE latent space...")
        Z_ae_umap = reduce_umap(Z_ae[tsne_idx])
        if Z_ae_umap is not None:
            plot_2d_embeddings(
                Z_ae_umap, ch5_labels[tsne_idx],
                "UMAP of AE Latent Space (Ch5 cluster labels)",
                "c6_05_umap_ae_clusters.png",
                legend_labels=["Cluster 0", "Cluster 1"],
            )
            plot_2d_embeddings(
                Z_ae_umap, ae_errors[tsne_idx],
                "UMAP of AE Latent Space (coloured by anomaly score)",
                "c6_05b_umap_ae_anomaly.png", cmap="plasma",
            )

    # t-SNE on raw features (reference comparison)
    log.info("  Running t-SNE on raw features (reference)...")
    Z_raw_tsne = reduce_tsne(X_pca[tsne_idx], perplexity=30)
    plot_2d_embeddings(
        Z_raw_tsne, ch5_labels[tsne_idx],
        "t-SNE of Raw Feature Space (Ch5 cluster labels)",
        "c6_07_tsne_raw_clusters.png",
        legend_labels=["Cluster 0", "Cluster 1"],
    )

    # (f) AE error vs Isolation Forest score
    plot_ae_vs_iso(ae_errors[tsne_idx], iso_scores[tsne_idx], rho_ae_iso)

    # ── 8. Anomaly case study ──────────────────────────────────────────
    log.info("\n── Top-10 anomaly case study ─────────────────────────────")
    top_anomaly_report(df, ae_errors, iso_scores, n=10)

    # ── 9. Summary comparison table ───────────────────────────────────
    summary = pd.DataFrame([
        {"Method": "AutoEncoder (AE)",
         "Latent dim": best_latent_dim,
         "Anomaly rate": round(float(anomalies_ae.mean()), 4),
         "p95 threshold": round(thr_ae, 5),
         "Sil (latent)": round(sil_ae_lat, 4),
         "Spearman vs ISO": round(rho_ae_iso, 3)},
        {"Method": "VAE (β-VAE)",
         "Latent dim": best_latent_dim,
         "Anomaly rate": round(float(anomalies_vae.mean()), 4),
         "p95 threshold": round(thr_vae, 5),
         "Sil (latent)": round(sil_vae_lat, 4),
         "Spearman vs ISO": round(rho_vae_iso, 3)},
        {"Method": "Isolation Forest",
         "Latent dim": "N/A",
         "Anomaly rate": round(float(anomalies_iso.mean()), 4),
         "p95 threshold": round(thr_iso, 5),
         "Sil (latent)": "N/A",
         "Spearman vs ISO": 1.0},
    ])
    summary.to_csv("results/c6_comparison_table.csv", index=False)

    log.info("\n" + "="*70)
    log.info("  COMPARISON TABLE")
    log.info("="*70)
    log.info(summary.to_string(index=False))
    log.info("="*70)

    log.info("\n  Silhouette scores (Ch5 labels as reference groups):")
    log.info(f"    Raw features: {sil_raw:.4f}")
    log.info(f"    AE latent:    {sil_ae_lat:.4f}")
    log.info(f"    VAE μ:        {sil_vae_lat:.4f}")

    log.info("\n✓ Challenge 6 complete.")
    log.info("  Figures: figures/c6_*")
    log.info("  Results: results/c6_*")
    log.info("  Models:  models/*.pt")


if __name__ == "__main__":
    main()