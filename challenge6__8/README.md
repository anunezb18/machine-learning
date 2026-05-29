# Challenge 6 — AutoEncoders & Representation Learning: EIA Energy Retail Sales

## 1. Overview

This project extends the unsupervised learning arc from Challenge 5 (clustering) into two advanced paradigms: **Anomaly Detection** via deep reconstruction models and **Representation Learning** with dimensionality-reduction visualisation. Three models are trained and compared on the same EIA electricity retail sales dataset used in Challenges 2 and 5:

- **AutoEncoder (AE)** — deterministic bottleneck network; reconstruction error as anomaly score.
- **Variational AutoEncoder (β-VAE)** — probabilistic latent space; μ vectors for downstream visualisation and clustering.
- **Isolation Forest** — classical tree-based baseline for cross-validation.

The final section of the paper synthesises findings across all three unsupervised challenges on the same domain.

---

## 2. Project Structure

```
challenge6__8/
├── challenge6.py             # Main script — all experiments
├── autoencoder.py            # AutoEncoder model definition
├── vae.py                    # VAE model definition + vae_loss()
├── eia_retail_sales.csv      # Dataset (same as Challenges 2 and 5)
├── requirements_c6.txt       # Python dependencies
├── CHECKLIST_c6.md           # Submission checklist
├── figures/                  # Generated plots
│   ├── c6_01_loss_curves.png
│   ├── c6_02_error_histograms.png
│   ├── c6_03_tsne_vae_anomaly.png
│   ├── c6_04_tsne_vae_clusters.png
│   ├── c6_05_umap_ae_clusters.png
│   ├── c6_05b_umap_ae_anomaly.png
│   ├── c6_06_ae_vs_iso.png
│   └── c6_07_tsne_raw_clusters.png
├── results/                  # Metric tables and anomaly reports
│   ├── c6_comparison_table.csv
│   ├── c6_ae_ablation.csv
│   ├── c6_vae_beta_sweep.csv
│   └── c6_top_anomalies.csv
├── models/                   # Saved model weights (.pt)
│   ├── ae_best.pt
│   ├── vae_best.pt
│   └── ...
└── logs/
    └── challenge6.log
```

---

## 3. Dataset

- **Source:** [EIA Electricity Data Browser](https://www.eia.gov/opendata/)
- **File:** `eia_retail_sales.csv`
- **Records after preprocessing:** 17,918 rows × 12 features
- **Period:** January 2001 → January 2026
- **Preprocessing:** identical to Challenge 5 (StandardScaler, no PCA before AE/VAE)
- **Features:** `sales`, `price`, `revenue_per_mwh`, `lag1`, `lag12`, `roll12_mean`, `roll12_std`, `yoy_growth`, `month_sin`, `month_cos`, `year_trend`, `state_enc`

---

## 4. Model Architectures

### AutoEncoder (AE)

| Parameter | Value |
|---|---|
| Input / Output dim | 12 |
| Hidden layers | [128, 64] |
| Latent dim | 16 *(best by Silhouette over {8, 16, 32})* |
| Activations | ReLU (hidden), Identity (output) |
| Loss | MSELoss |
| Epochs | 100 |
| Learning rate | 1e-3 |
| Batch size | 256 |
| Optimizer | Adam |

### Variational AutoEncoder (β-VAE)

| Parameter | Value |
|---|---|
| Input / Output dim | 12 |
| Hidden layers | [128, 64] |
| Latent dim | 16 *(same as AE)* |
| β | 4.0 *(best by Silhouette over {0.5, 1.0, 4.0})* |
| KL warm-up | 20 epochs (0 → β) |
| Epochs | 100 |
| Learning rate | 1e-3 |
| Batch size | 256 |
| Optimizer | Adam |

---

## 5. Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements_c6.txt

# 3. Run all experiments
python challenge6.py
```

> **Note:** `umap-learn` must be installed separately — it is not part of scikit-learn.
> It is included in `requirements_c6.txt`, but if you encounter import errors run:
> `pip install umap-learn`

---

## 6. Reproducing Results

All random seeds are fixed (`SEED = 42`). Running `challenge6.py` will:

1. Load and preprocess `eia_retail_sales.csv` (identical pipeline to Challenge 5)
2. Perform a stratified 80/20 train/test split using Challenge 5 cluster labels
3. Fit an **Isolation Forest** baseline (`contamination=0.05`, 200 estimators)
4. Run **AE architecture ablation** over latent dims {8, 16, 32}; select best by Silhouette
5. Run **AE multi-seed stability** check with seeds [42, 123, 777]
6. Run **VAE β-sweep** over {0.5, 1.0, 4.0} with KL warm-up; select best by Silhouette
7. Run **VAE multi-seed stability** check with seeds [42, 123, 777]
8. Compute **Spearman ρ** between all three detector rankings
9. Generate **t-SNE** (VAE latent space + raw features) and **UMAP** (AE latent space) visualisations
10. Save all figures to `figures/` and metrics to `results/`

---

## 7. Results Summary

### Anomaly Detection

| Method | Latent dim | Anomaly rate | p95 threshold | Spearman vs ISO |
|---|---|---|---|---|
| AutoEncoder (AE) | 16 | 5.00% | 0.00059 | ρ = 0.650 |
| VAE (β = 4.0) | 16 | 5.00% | 1.06603 | ρ = 0.760 |
| Isolation Forest | — | 4.97% | 0.56815 | ρ = 1.000 |

### Representation Quality (Silhouette Score)

Using Challenge 5 K-Means cluster labels as reference groups:

| Space | Silhouette |
|---|---|
| Raw features (PCA-10) | 0.7808 |
| AE latent space | 0.8102 |
| VAE μ vectors | **0.8893** |

The VAE learned a significantly more discriminative representation than both raw features and the deterministic AE.

### Multi-Seed Stability

| Model | p95 threshold (mean ± std) |
|---|---|
| AE | 0.0007 ± 0.0001 |
| VAE | 1.0315 ± 0.0266 |

Both models are stable across seeds [42, 123, 777].

---

## 8. Top Anomalies (AE Error)

All 10 highest-error observations correspond to `U.S. Total` rows — national aggregate records whose `sales` values are orders of magnitude larger than any individual state. These are structural outliers that K-Means silently absorbed into the large cluster in Challenge 5.

| Period | State | Sales (MWh) | AE Error | ISO Score |
|---|---|---|---|---|
| 2026-01 | U.S. Total | 355,940 | 0.0076 | 0.762 |
| 2011-01 | U.S. Total | 334,116 | 0.0073 | 0.745 |
| 2025-02 | U.S. Total | 320,561 | 0.0065 | 0.757 |
| 2002-01 | U.S. Total | 290,967 | 0.0063 | 0.751 |
| 2011-12 | U.S. Total | 301,844 | 0.0062 | 0.733 |

---

## 9. Cross-Challenge Synthesis

| Challenge | Paradigm | Key finding |
|---|---|---|
| Challenge 2 | Semi-supervised | 10% labeled data is enough for 92% accuracy — the consumption signal is strong and structured |
| Challenge 5 | Clustering | Dominant structure is scale-based; mega-states (TX, CA) separate cleanly; DBSCAN flagged HI and AK as noise |
| Challenge 6 | Deep representation | AE/VAE improve cluster separability (Silhouette +0.03 / +0.11); U.S. Total rows are the true anomalies; AE–VAE ρ = 0.509 shows the two detectors are complementary |

---

## 10. Seeds Used

| Experiment | Seeds |
|---|---|
| Main run | 42 |
| AE multi-seed stability | 42, 123, 777 |
| VAE multi-seed stability | 42, 123, 777 |

---

### Authors

- **Laura Sofia Culma Ospina** — lsculmao@udistrital.edu.co
- **Alejandro Nuñez Barrera** — anunezb@udistrital.edu.co