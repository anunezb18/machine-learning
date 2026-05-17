# Challenge 5 — Clustering Checklist

---

## Dataset

| Field | Value |
|---|---|
| **Name** | EIA Retail Sales of Electricity |
| **Source URL** | https://www.eia.gov/electricity/data/browser/ |
| **Raw records** | 111,972 |
| **Records after preprocessing** | 17,918 (ALL-sectors aggregate, 62 states, 2001-01 → 2026-01) |
| **Features used** | 8 (`yoy_growth`, `roll12_std`, `roll12_mean`, `price`, `month_sin`, `month_cos`, `year_trend`, `revenue_per_mwh`) |
| **PCA components** | 5 (retaining 90.8% of variance) |

---

## Final Hyperparameter Configurations

### K-Means
| Hyperparameter | Value |
|---|---|
| `n_clusters` (k) | 2 |
| `init` | k-means++ |
| `n_init` | 10 |
| `max_iter` | 300 |
| `random_state` | 42 |

### DBSCAN
| Hyperparameter | Value |
|---|---|
| `eps` | 1.093 |
| `min_samples` | 10 |
| Clusters found | 2 |
| Noise fraction | 0.43% (77 points) |

### Hierarchical (Agglomerative)
| Hyperparameter | Value |
|---|---|
| `linkage` | ward |
| `n_clusters` | 2 |
| Dendrogram sample size | 2,000 |

---

## Evaluation Metrics (Best Configuration per Algorithm)

| Algorithm | Clusters | Noise % | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ |
|---|---|---|---|---|---|
| K-Means (k=2) | 2 | 0.00% | **0.7162** | 0.3965 | **5634.4** |
| DBSCAN (eps=1.093, ms=10) | 2 | 0.43% | **0.7207** | **0.3898** | 5428.3 |
| Hierarchical (ward, k=2) | 2 | 0.00% | 0.7162 | 0.3965 | 5634.4 |

---

## Algorithm Comparison and Recommendation

All three algorithms converge on the same two-cluster solution for this dataset, which reflects a dominant seasonal split in U.S. electricity retail sales: a warm-season cluster (high consumption, high price volatility) and a cold-season cluster (lower consumption, greater stability). K-Means and Hierarchical (Ward linkage) produce identical partitions with Silhouette = 0.716 and CH = 5634.4, confirming that the structure is globally convex and well-separated. DBSCAN with eps = 1.093 and min\_samples = 10 achieves a marginally better Silhouette (0.721) and Davies–Bouldin (0.390) while flagging 77 anomalous observations (0.4% noise), which represent states with atypical consumption spikes or data anomalies — a domain-relevant finding that the other two algorithms absorb silently into clusters. **DBSCAN is the recommended algorithm** for this dataset: it matches the quantitative performance of K-Means and Hierarchical while uniquely identifying outlier state-months of anomalous energy consumption, directly addressing the domain question posed in Challenge 5 for Group 8.