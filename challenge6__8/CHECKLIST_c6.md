# CHECKLIST — Challenge 6 / Group 8 / EIA Energy Clustering

## Dataset
- **Name:** EIA Electricity Retail Sales — Monthly by State and Sector
- **Source:** https://www.eia.gov/opendata/
- **File:** `eia_retail_sales.csv` (same as Challenges 2 and 5)
- **Records after preprocessing:** ~17,918 rows × 12 features
- **Preprocessing:** identical to Challenge 5 (StandardScaler, no PCA before AE/VAE)

## Final Model Architectures

| Parameter | AE | VAE |
|---|---|---|
| Input dim | 12 | 12 |
| Hidden dims | [128, 64] | [128, 64] |
| Latent dim | 16 (best by Silhouette over {8, 16, 32}) | 16 (same as AE) |
| Activation | ReLU (hidden), Identity (output) | ReLU (hidden), Identity (output) |
| Output activation | None | None |
| Epochs | 100 | 100 |
| Learning rate | 1e-3 | 1e-3 |
| Batch size | 256 | 256 |
| β (VAE) | — | 4.0 (best by Silhouette over {0.5, 1.0, 4.0}) |
| KL warm-up | — | 20 epochs (0 → β) |
| Optimizer | Adam | Adam |

## Anomaly Thresholds (fill after running)

| Method | Threshold | Anomaly Rate |
|---|---|---|
| AE (p95) | 0.00059 | 5.0% |
| VAE (p95) | 1.06603 | 5.0% |
| Isolation Forest (internal offset) | 0.56815 | 4.97% |

## Spearman Rank Correlations

| Pair | ρ |
|---|---|
| AE vs Isolation Forest | 0.650 |
| VAE vs Isolation Forest | 0.760 |
| AE vs VAE | 0.509 |

## Silhouette Scores (Ch5 cluster labels as reference)

| Space | Silhouette |
|---|---|
| Raw features (PCA-10) | 0.7808 |
| AE latent space | 0.8102 |
| VAE μ vectors | 0.8893 |

## Seeds Used
- All experiments: **42**
- Multi-seed stability check (AE): **42, 123, 777** → p95 threshold: 0.0007 ± 0.0001
- Multi-seed stability check (VAE): **42, 123, 777** → p95 threshold: 1.0315 ± 0.0266

## Commands to Reproduce
```bash
pip install -r requirements_c6.txt

python challenge6.py
```

## Cross-Challenge Summary (≤200 words)

Challenge 2 (semi-supervised) showed that 10% labeled EIA data is enough
to classify high-consumption months with 92% accuracy using Random Forest,
suggesting that the consumption signal is strong and structured.

Challenge 5 (clustering) revealed that the dominant structure in the data
is scale-based: two clusters separate the handful of mega-states (Texas,
California) from the rest. DBSCAN added value by flagging month-state
observations with anomalously high prices (Hawaii, Alaska) as noise,
a finding K-Means absorbed silently into the larger cluster.

Challenge 6 (autoencoders) goes deeper. The AE (latent=16) and VAE (β=4.0)
both improve cluster separability over raw features: Silhouette rises from
0.7808 (raw) to 0.8102 (AE) and 0.8893 (VAE), confirming that deep models
learn more discriminative representations. The AE reconstruction error flags
national aggregate rows (U.S. Total) as the most anomalous observations —
values orders of magnitude larger than any individual state — a structural
outlier that K-Means silently absorbed into the large cluster. The moderate
AE–VAE Spearman correlation (ρ=0.509) shows the two detectors are
complementary: the VAE's probabilistic latent space captures regularity
patterns that the deterministic AE misses. Together, the three challenges
form a complete unsupervised picture of U.S. electricity consumption structure.