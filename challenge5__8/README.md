# Challenge 5 — Unsupervised Clustering: EIA Energy Retail Sales

## 1. Overview

This project applies three unsupervised clustering algorithms — **K-Means**, **DBSCAN**, and **Hierarchical (Agglomerative)** — to U.S. electricity retail sales data from the EIA (Energy Information Administration). The goal is to discover latent consumption patterns across U.S. states from 2001 to 2026, with a focus on identifying anomalous consumption outliers.

## 2. Project Structure

```
challenge5__8/
├── challenge5.ipynb          # Main notebook — all experiments
├── eia_retail_sales.csv      # Dataset (EIA retail sales)
├── requirements.txt          # Python dependencies
├── CHECKLIST.md              # Submission checklist
├── figures/                  # Generated plots
│   ├── 00_pca_variance.png
│   ├── 01_kmeans_elbow_silhouette.png
│   ├── 02_dbscan_knn_distance.png
│   ├── 03_hierarchical_dendrogram.png
│   ├── 04_kmeans_clusters.png
│   ├── 05_dbscan_clusters.png
│   ├── 06_hierarchical_clusters.png
│   └── 07_kmeans_temporal_subset.png
├── results/                  # Metric tables and cluster profiles
│   ├── comparison_table.csv
│   ├── comparison_table_full.csv
│   ├── dbscan_sweep.csv
│   ├── hierarchical_sweep.csv
│   ├── kmeans_profiles.csv
│   ├── dbscan_profiles.csv
│   └── hierarchical_profiles.csv
└── logs/
    └── challenge5.log
```

## 3. Dataset

- **Source:** [EIA Electricity Data Browser](https://www.eia.gov/electricity/data/browser/)
- **File:** `eia_retail_sales.csv`
- **Records after preprocessing:** 17,918 (ALL-sectors aggregate, 62 states)
- **Period:** January 2001 → January 2026
- **Features:** `yoy_growth`, `roll12_std`, `roll12_mean`, `price`, `month_sin`, `month_cos`, `year_trend`, `revenue_per_mwh`

## 4. Setup

```bash

# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook challenge5.ipynb
```

## 5. Reproducing Results

All random seeds are fixed (`random_state=42`). Run all cells in `challenge5.ipynb` top to bottom. The notebook will:

1. Load and preprocess `eia_retail_sales.csv`
2. Apply PCA (5 components, 90.8% variance retained)
3. Run K-Means sweep (k ∈ {2..12}), select best k by Silhouette
4. Run DBSCAN sweep (eps × min_samples grid), select best by composite score
5. Run Hierarchical clustering (ward, complete, average linkage comparison)
6. Run feature ablation (temporal subset vs. economic subset)
7. Save all figures to `figures/` and metrics to `results/`

## 6. Results Summary

| Algorithm | k | Silhouette | Davies–Bouldin | Calinski–Harabasz |
|---|---|---|---|---|
| K-Means | 2 | 0.716 | 0.397 | 5634.4 |
| DBSCAN | 2 | **0.721** | **0.390** | 5428.3 |
| Hierarchical (ward) | 2 | 0.716 | 0.397 | 5634.4 |

**Recommended algorithm:** DBSCAN — matches K-Means and Hierarchical quantitatively while uniquely identifying 77 anomalous state-month observations (0.4% noise).
