# **Machine Learning Class Repository**
This repository contains materials and implementations developed mainly for a Machine Learning class. It includes practical all challenges proposed during the course.

## Project Index
* **[Challenge 1: Atari Phoenix](./challenge1__8/):** Training a **Deep Q-Network (DQN)** agent.
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

## Quick Start
Each project folder is self-contained. To replicate or test the results:
1.  Navigate to the specific challenge folder: `cd challenge_1`
2.  Install required dependencies: `pip install -r requirements.txt`
3.  Run the main script: `python phoenix.py --mode play --model-path models/phoenix_best`

### Authors
- **Laura Sofia Culma Ospina** - lsculmao@udistrital.edu.co
- **Alejandro Nuñez Barrera** - anunezb@udistrital.edu.co
