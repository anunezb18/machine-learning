
# Deep Reinforcement Learning with DQN on ALE/Phoenix-v5

## 1. Overview

This repository contains an implementation of a Deep Q-Network (DQN) agent trained on the Atari environment *ALE/Phoenix-v5* using Stable-Baselines3. The project includes hyperparameter sweeps, reproducible experiments using multiple random seeds, and logging through TensorBoard.

---

## 2. Requirements

### System

* Linux (Ubuntu 20.04+ recommended)
* Python 3.11

### Dependencies

Install the required packages:

```bash
pip install stable-baselines3[extra]
pip install gymnasium[atari]
pip install ale-py
pip install tensorboard
pip install numpy
```

---

Or use the requirements.txt

```bash
pip install -r requirements.txt
```

---

## 3. Environment Setup (Linux)

Create and activate a virtual environment:

```bash
python3 -m venv dqn_env
source dqn_env/bin/activate
```

Upgrade pip:

```bash
pip install --upgrade pip
```

Install dependencies (see section above).

---

## 4. Project Structure

```text
.
├── phoenix.py
├── sweep_configs.json
├── models/
│   └── _sweep_tmp/
├── logs/
│   └── phoenix_dqn/
└── README.md
```

---

## 5. Running Experiments

### 5.1 Training a Single Model

```bash
python phoenix.py --mode train \
  --model-path models/phoenix \
  --timesteps 300000 \
  --tensorboard-log logs/phoenix_dqn \
  --seed 42
```

---

### 5.2 Running Hyperparameter Sweep

```bash
python phoenix.py --mode sweep \
  --sweep-file sweep_configs.json \
  --model-path models/phoenix \
  --timesteps 300000 \
  --tensorboard-log logs/phoenix_dqn
```

This will:

* Execute all configurations in the JSON file
* Train each configuration using three seeds (42, 7, 99)
* Store models in `models/_sweep_tmp/`
* Log results in `logs/phoenix_dqn/sweep/`

---

### 5.3 Playing a Trained Model

```bash
python phoenix.py --mode play \
  --model-path models/_sweep_tmp/exp_01_s42 \
  --episodes 5
```

---

## 6. TensorBoard Visualization

To visualize training logs:

```bash
tensorboard --logdir logs/phoenix_dqn/sweep --port 6006
```

Then open in your browser:

```text
http://localhost:6006
```

Logs are organized as:

```text
logs/phoenix_dqn/sweep/
  ├── exp_01/
  │   ├── seed_42/
  │   ├── seed_7/
  │   └── seed_99/
```

This structure allows comparison across seeds and visualization of variance.

---

## 7. Reproducibility

All experiments are reproducible due to:

* Fixed random seeds: 42, 7, 99
* Deterministic hyperparameter configurations defined in `sweep_configs.json`
* Controlled training budget (300,000 timesteps per experiment)

To reproduce results:

1. Set up the environment
2. Run the sweep command
3. Visualize results in TensorBoard

---

## 8. Logging Artifacts

Sample TensorBoard logs are included in the `logs/` directory. These logs contain:

* Episode rewards
* Exploration rate (epsilon)
* Training progress

They can be directly visualized without retraining.

---

## 9. Notes

* Training is CPU-based and may take several hours depending on the number of experiments.
* Ensure sufficient disk space for logs and model checkpoints.
* It is recommended to keep the system connected to power during training.