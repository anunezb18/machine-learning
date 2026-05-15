# ALE/Phoenix-v5 — DQN · PPO · BC · GAIL

Comparative study of four deep reinforcement / imitation learning algorithms on the **ALE/Phoenix-v5** Atari environment (Challenge 4, Group 8 — Universidad Distrital Francisco José de Caldas).

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Workflow](#workflow)
  - [1. Collect Demonstrations](#1-collect-demonstrations)
  - [2. Train Behavioral Cloning](#2-train-behavioral-cloning)
  - [3. Train GAIL](#3-train-gail)
  - [4. Run a Sweep](#4-run-a-sweep)
  - [5. Evaluate a Checkpoint](#5-evaluate-a-checkpoint)
  - [6. Watch the Agent Play](#6-watch-the-agent-play)
- [TensorBoard](#tensorboard)
- [Sweep Configuration](#sweep-configuration)
- [Algorithm Details](#algorithm-details)
- [Results Summary](#results-summary)

---

## Overview

| Algorithm | Paradigm | Env reward used | Notes |
|-----------|----------|-----------------|-------|
| DQN | Value-based | Yes | SB3, replay buffer |
| PPO | Policy-gradient | Yes | Custom PyTorch, on-policy |
| BC | Supervised imitation | No | Cross-entropy on demos |
| GAIL | Adversarial imitation | No (+ 0 % blend) | PPO + discriminator, BC warm-start |

All agents use the **same** convolutional backbone (3 conv layers → 3136-dim feature vector) and the **same** preprocessing pipeline (84×84 grayscale, 4-frame stack, frame-skip 4, up to 30 no-ops at episode start).

---

## Project Structure

```
.
├── collect_demos.py          # Record (obs, action) pairs from a trained checkpoint
├── train_bc.py               # Behavioral Cloning training & evaluation
├── train_gail.py             # GAIL training, evaluation, play, and sweep
├── sweep_configs_gail.json   # Sweep configurations for GAIL experiments
├── requirements.txt          # Python dependencies
├── demos/                    # Demonstration datasets (created at runtime)
│   ├── demos_dqn_20k.npz
│   ├── demos_dqn_20k_info.json
│   ├── demos_dqn_5k.npz
│   └── demos_dqn_5k_info.json
├── models/                   # Saved checkpoints (created at runtime)
│   ├── phoenix_bc.pt
│   ├── phoenix_gail.pt
│   └── best_gail.pt
└── logs/                     # TensorBoard logs (created at runtime)
    ├── phoenix_bc/
    └── phoenix_gail/
```

---

## Requirements

```
numpy<2
torch
tensorboard
gymnasium[atari,accept-rom-license]>=0.29.1,<1.1.0
ale-py==0.10.1
autorom[accept-rom-license]
opencv-python==4.8.1.78
stable-baselines3[extra]>=2.3,<3
tqdm
```

> **CPU-only training** is fully supported. All experiments in the paper ran without GPU acceleration.

---

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download Atari ROMs (required once)
autorom --accept-license
```

---

## Workflow

### 1. Collect Demonstrations

Record expert trajectories from a pre-trained DQN checkpoint (`.zip` from Stable-Baselines3):

```bash
# Large dataset — 20 000 steps (~12–18 episodes)
python collect_demos.py \
    --checkpoint models/phoenix_best.zip \
    --source dqn \
    --n-steps 20000 \
    --seed 42 \
    --out demos/demos_dqn_20k.npz

# Small dataset — 5 000 steps (~3–5 episodes)
python collect_demos.py \
    --checkpoint models/phoenix_best.zip \
    --source dqn \
    --n-steps 5000 \
    --seed 42 \
    --out demos/demos_dqn_5k.npz
```

A `_info.json` metadata file is written alongside each `.npz` with episode statistics and demonstrator return.

> `--source ppo` is also accepted for custom PyTorch PPO checkpoints (`.pt`).

---

### 2. Train Behavioral Cloning

BC trains a supervised policy on the recorded demonstrations. The best checkpoint is selected by real evaluation return (not training loss).

```bash
python train_bc.py \
    --mode train \
    --demos demos/demos_dqn_20k.npz \
    --model-path models/phoenix_bc \
    --n-epochs 100 \
    --batch-size 256 \
    --lr 3e-4 \
    --seed 42 \
    --tensorboard-log logs/phoenix_bc
```

Key training details:
- Adam optimiser with weight decay `1e-5`
- Dropout 0.2 in actor and critic heads
- Light data augmentation: additive Gaussian noise clamped to `[0, 1]`
- Model saved whenever real evaluation return improves (every 5 epochs, 3 episodes)

---

### 3. Train GAIL

GAIL uses the BC checkpoint as a warm-start and then trains the policy adversarially against a discriminator.

```bash
python train_gail.py \
    --mode train \
    --demos demos/demos_dqn_20k.npz \
    --model-path models/phoenix_gail \
    --timesteps 300000 \
    --seed 42 \
    --use-action false \
    --bc-init models/phoenix_bc \
    --tensorboard-log logs/phoenix_gail
```

| Flag | Description | Default |
|------|-------------|---------|
| `--use-action` | Include one-hot action in discriminator input (GAIL-B) | `false` |
| `--bc-init` | Path to BC checkpoint for warm-start (without `.pt`) | `models/phoenix_bc` |
| `--timesteps` | Environment steps budget | `300000` |
| `--seed` | Random seed | `42` |

---

### 4. Run a Sweep

Train all GAIL configurations defined in `sweep_configs_gail.json` across seeds 42, 7, and 99. The best model is automatically copied to `models/best_gail.pt`.

```bash
python train_gail.py \
    --mode sweep \
    --sweep-file sweep_configs_gail.json \
    --tensorboard-log logs/phoenix_gail \
    --bc-init models/phoenix_bc
```

The sweep covers three experiments:

| Name | Discriminator input | Demo size |
|------|---------------------|-----------|
| `gail_A_20k` | State only | 20 000 |
| `gail_A_5k` | State only | 5 000 |
| `gail_B_20k` | State + action | 20 000 |

---

### 5. Evaluate a Checkpoint

Measure mean episodic return over a fixed number of episodes using the saved model:

```bash
# BC
python train_bc.py \
    --mode eval \
    --model-path models/phoenix_bc \
    --episodes 10 \
    --seed 42

# GAIL
python train_gail.py \
    --mode eval \
    --model-path models/phoenix_gail \
    --episodes 10 \
    --seed 42
```

---

### 6. Watch the Agent Play

Render the environment in a window to visually inspect the trained policy:

```bash
python train_gail.py \
    --mode play \
    --model-path models/best_gail \
    --episodes 3
```

> Requires a display. On headless servers use `xvfb-run` or set `DISPLAY=:0`.

---

## TensorBoard

All training runs log to the `logs/` directory. Launch TensorBoard with:

```bash
tensorboard --logdir logs/
```

Logged scalars include:

| Tag | Meaning |
|-----|---------|
| `gail/episode_reward_real` | True environment return per episode |
| `gail/disc_loss` | Discriminator BCE loss per rollout |
| `gail/disc_accuracy` | Fraction of expert/agent samples correctly classified |
| `gail/adv_reward_mean` | Mean adversarial reward over the rollout |
| `bc/epoch_loss` | Cross-entropy loss per epoch |
| `bc/epoch_accuracy` | Cloning accuracy per epoch |
| `bc/eval_return` | Real return during periodic BC evaluation |

---

## Sweep Configuration

`sweep_configs_gail.json` is a JSON array where each object defines one GAIL experiment. All keys are optional and fall back to the defaults in `train_gail.py`.

```json
[
  {
    "name": "gail_A_20k",
    "demos_path": "demos/demos_dqn_20k.npz",
    "use_action": false,
    "timesteps": 300000,
    "horizon": 512,
    "n_ppo_epochs": 10,
    "batch_size": 128,
    "lr_policy": 0.00025,
    "lr_disc": 0.00001,
    "disc_updates": 1,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5
  }
]
```

---

## Algorithm Details

### Shared CNN Backbone

```
Conv2d(4 → 32, kernel=8, stride=4)  + ReLU
Conv2d(32 → 64, kernel=4, stride=2) + ReLU
Conv2d(64 → 64, kernel=3, stride=1) + ReLU
Flatten → 3136-dim vector
```

### GAIL Discriminator

- Same CNN backbone as the policy
- FC(3136 [+ n_actions] → 512, Tanh) → FC(512 → 1)
- Trained with `BCEWithLogitsLoss` and label smoothing (expert=0.9, agent=0.1)
- GAIL-B concatenates the one-hot action to the CNN features before the FC layer

### Adversarial Reward

The policy is updated with PPO using the reward signal `−log(1 − D(s,a))`, derived from the discriminator output before sigmoid. This non-saturating formulation provides stronger gradients early in training.

### BC Warm-Start

Before GAIL training begins, the policy is initialised from the best BC checkpoint. This reduces the number of environment steps needed for the discriminator to produce a useful reward signal.

---

## Results Summary

| Algorithm | Config | Mean return ± std |
|-----------|--------|-------------------|
| DQN | Aggressive | 3 324.3 ± 95.2 |
| PPO | Aggressive | 757.5 ± 52.7 |
| BC | 20k demos | — |
| GAIL-A | 20k demos | — |
| GAIL-A | 5k demos | — |
| GAIL-B | 20k demos | — |

> Results for BC and GAIL will be filled in after experimental runs complete.

All scores are mean episodic return over the last 50 episodes, averaged across seeds 42, 7, and 99.
