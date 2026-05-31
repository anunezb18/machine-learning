# **Machine Learning Class Repository**
This repository contains materials and implementations developed mainly for a Machine Learning class. It includes practical all challenges proposed during the course.
 
## Project Index
* **[Challenge 1: Atari Phoenix](./challenge1__8/):** Training a **Deep Q-Network (DQN)** agent on the Atari environment *ALE/Phoenix-v5*.
    * **Achievement:** Reached a high score of **4,850 points** (Expert-level performance).
    * **Tech Stack:** Stable-Baselines3, Gymnasium, TensorBoard.
* **[Challenge 2: Learning with Limited Labels](./challenge2__8/):** Semi-Supervised Learning for electricity consumption classification.
    * **Approach:** Self-training using pseudo-labels with confidence thresholding.
    * **Achievement:** Improved **Recall by 4.4%** with only **10% labeled data**.
    * **Models:** Logistic Regression, Random Forest, Self-training SSL.
    * **Tech Stack:** Scikit-learn, NumPy, Pandas.
* **[Challenge 3: PPO vs DQN](./challenge3__8/):** Comparison between **Proximal Policy Optimization (PPO)** and **Deep Q-Network (DQN)** on the Atari environment *ALE/Phoenix-v5*.
    * **Objective:** Evaluate which algorithm performs better under a fixed computational budget.
    * **Focus:** Sample efficiency, training stability, and final performance.
    * **Preliminary Result:** DQN achieved better performance, while PPO showed more stable but slower learning.
    * **Tech Stack:** PyTorch, Gymnasium, Stable-Baselines3, TensorBoard.
* **[Challenge 4: DQN, PPO and GAIL Comparison](./challenge4__8/):** Full three-algorithm comparison extending Challenge 3 by introducing **Generative Adversarial Imitation Learning (GAIL)** and a **Behavioral Cloning (BC)** baseline on *ALE/Phoenix-v5*.
    * **Objective:** Evaluate whether an agent that learns by imitating DQN demonstrations through adversarial training can outperform BC and approach DQN/PPO under the same 300,000-step budget.
    * **Ablations:** State-only vs. state+action discriminator; 5k vs. 20k demonstration dataset size.
    * **Tech Stack:** PyTorch, Gymnasium, Stable-Baselines3, TensorBoard, Matplotlib.
* **[Challenge 5: Unsupervised Clustering](./challenge5__8/):** Clustering analysis of U.S. electricity retail sales data from the EIA using **K-Means**, **DBSCAN**, and **Hierarchical (Agglomerative)** clustering on the same Energy & Utilities domain as Challenge 2.
    * **Objective:** Discover latent consumption patterns across U.S. states and identify anomalous energy consumption outliers without using any labels.
    * **Ablations:** Full feature set vs. temporal subset vs. economic subset; Ward vs. complete vs. average linkage.
    * **Tech Stack:** Scikit-learn, NumPy, Pandas, Matplotlib, SciPy.
* **[Challenge 6: AutoEncoders & Representation Learning](./challenge6__8/):** Anomaly detection and representation learning on the EIA electricity retail sales dataset using deep generative models, closing the unsupervised learning arc started in Challenges 2 and 5.
    * **Objective:** Detect anomalous consumption records and learn discriminative latent representations without labels, then synthesise findings across all three unsupervised challenges.
    * **Models:** AutoEncoder (AE), Variational AutoEncoder (β-VAE), Isolation Forest baseline.
    * **Achievements:** Silhouette Score improved from **0.7808** (raw features) to **0.8893** (VAE μ vectors); top anomalies identified as U.S. national aggregate rows — structural outliers silently absorbed by K-Means in Challenge 5.
    * **Ablations:** Latent dimension ∈ {8, 16, 32}; β ∈ {0.5, 1.0, 4.0}; multi-seed stability across seeds [42, 123, 777].
    * **Tech Stack:** PyTorch, Scikit-learn, NumPy, Pandas, Matplotlib, umap-learn.
* **[Challenge 7: Transfer Learning & Domain Adaptation](./challenge7__8/):** Transfer learning, neural style transfer, and domain adaptation for digit recognition under severe distribution shift between SVHN street-view photographs and MNIST handwritten digits.
    * **Objective:** Evaluate how pretrained representations, synthetic style-transfer augmentation, and domain adaptation techniques mitigate performance degradation caused by cross-domain visual discrepancies.
    * **Models:** ResNet-50 (Frozen Backbone, Fine-Tuning, From Scratch), Neural Style Transfer (VGG-19), Domain-Adversarial Neural Network (DANN).
    * **Achievements:** Target-domain accuracy improved from **41.6%** (unadapted baseline) to **93.9%** with supervised adaptation; DANN achieved **63.9%** accuracy without target labels, while style-transfer augmentation improved performance to **50.6%** using only synthetic target-style data.
    * **Ablations:** Transfer-learning strategy comparison (Frozen vs. Fine-Tuned vs. Scratch), adaptation strategy comparison (Baseline, Style-Aug, DANN, Target Fine-Tune), and multi-seed evaluation across seeds **[42, 123, 777]**.
    * **Tech Stack:** PyTorch, Torchvision, Scikit-learn, NumPy, Pandas, Matplotlib, Grad-CAM, t-SNE.

## Quick Start
Each project folder is self-contained. To replicate or test the results:
1.  Navigate to the specific challenge folder: `cd challenge_1`
2.  Install required dependencies: `pip install -r requirements.txt`
3.  Run the main script: `python phoenix.py --mode play --model-path models/phoenix_best`

### Authors
- **Laura Sofia Culma Ospina** - lsculmao@udistrital.edu.co
- **Alejandro Nuñez Barrera** - anunezb@udistrital.edu.co