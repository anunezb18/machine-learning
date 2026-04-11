# Challenge 2 — Semi-Supervised Learning with Limited Labels

## 1. Overview
This repository contains the implementation of a **Semi-Supervised Learning (SSL)** pipeline using **Self-Training** to classify high electricity consumption periods. Using real-world data from the U.S. Energy Information Administration (EIA), the project leverages a large unlabeled pool (90%) to enhance the performance of a Random Forest baseline trained on limited labeled data (10%).

---

## 2. Key Contributions
- Semi-supervised self-training pipeline
- Strict reproducibility protocol (multi-seed evaluation)
- No look-ahead bias temporal feature engineering
- Confidence-based pseudo-label filtering
- Full experimental logging and metrics tracking

---

## 3. Requirements

### System
- Python 3.11 recommended
- Linux / macOS / Windows

### Dependencies
Install the required libraries:

```bash
pip install -r requirements.txt
```

Core dependencies: `scikit-learn`, `pandas`, `numpy`, `matplotlib`

---

## 4. Environment Setup
It is recommended to use a virtual environment:

```bash
python3 -m venv ssl_env
source ssl_env/bin/activate  # On Windows: ssl_env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Project Structure
```
.
├── challenge2.ipynb
├── eia_retail_sales.csv
├── requirements.txt
├── logs/
│   └── challenge2.log
├── results/
│   ├── challenge2_summary.csv
│   └── challenge2_detail.csv
```

---

## 6. Reproducibility Protocol
To ensure scientific reproducibility, the following protocol was used:

- Fixed seeds: `42`, `123`, `777`
- Strict 20% held-out test set
- No look-ahead bias feature engineering
- Confidence threshold: `0.75`
- Identical preprocessing across runs

---

## 7. Running Experiments

1. Place `eia_retail_sales.csv` in the root directory  
2. Execute all cells in `challenge2.ipynb`  

The notebook will automatically:
- Perform temporal feature engineering
- Train supervised baselines (Logistic Regression, Random Forest)
- Execute SSL Self-Training loop
- Save metrics and logs

---

## 8. Results Summary

| Model | F1-macro | AUC | Recall (Class 1) |
|------|----------|-----|------------------|
| Logistic Regression | 0.795 | 0.876 | 0.791 |
| Random Forest | **0.922** | **0.981** | 0.859 |
| **SSL Random Forest** | 0.913 | 0.970 | **0.903** |

**Key Result:** Semi-supervised learning improves Recall by **+4.4%** under limited labeled data.

---

## 9. Artifacts

- Logs: `logs/challenge2.log`
- Metrics: `results/challenge2_detail.csv`
- Aggregated results: `results/challenge2_summary.csv`