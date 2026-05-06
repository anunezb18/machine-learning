# Deep Reinforcement Learning with PPO on ALE/Phoenix-v5

## 1. Overview

This folder contains an implementation of a **Proximal Policy Optimization (PPO)** agent trained on the Atari environment *ALE/Phoenix-v5*. This project is a **direct extension of Challenge 1**, where the same environment was tackled using a Deep Q-Network (DQN). Challenge 3 implements PPO with clipped surrogate objective, Generalised Advantage Estimation (GAE), and a shared convolutional Actor-Critic network, then produces a side-by-side empirical comparison against the DQN results from Challenge 1.

The central research question addressed is:

> *"Under a fixed computational budget and on the same environment, does PPO converge faster, reach higher performance, or exhibit different failure modes compared to the DQN agent from Challenge 1? Why?"*

---

## 2. Requirements

### System

- Python 3.11+
- Windows / Linux / macOS

### Dependencies

Install required packages:

```bash
pip install torch torchvision
pip install gymnasium[atari]
pip install ale-py
pip install numpy
pip install tensorboard
pip install opencv-python
pip install tqdm
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

---

## 3. Project Structure

```text
challenge3/
├── challenge3.py           # Main PPO agent script
├── phoenix.py              # DQN agent script (Challenge 1)
├── sweep_configs_ppo.json  # PPO hyperparameter configurations
├── sweep_configs.json      # DQN hyperparameter configurations (Challenge 1)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── CHECKLIST.md            # Reproducibility checklist
├── models/
│   ├── _sweep_ppo_tmp/     # PPO model checkpoints (.pt)
│   └── _sweep_tmp/         # DQN model checkpoints (.zip)
└── logs/
    ├── phoenix_ppo/        # PPO TensorBoard logs
    │   └── sweep/
    │       ├── exp_01/
    │       ├── exp_02/
    │       └── exp_03/
    └── phoenix_dqn/        # DQN TensorBoard logs (Challenge 1)
        └── sweep/
```

---

## 4. PPO Algorithm — Implemented Components

- On-policy rollout collection for T environment steps (horizon)
- Generalised Advantage Estimation (GAE) with parameters γ and λ
- Clipped surrogate objective: `L_CLIP = E[min(r_t(θ) * A_t, clip(r_t(θ), 1−ε, 1+ε) * A_t)]`
- Value-function loss: `L_VF = E[(V_θ(s_t) − R_t)²]`
- Entropy bonus: `L_ENT = E[H(π_θ(·|s_t))]`
- Combined loss: `L = −L_CLIP + c1 * L_VF − c2 * L_ENT`
- Multiple mini-batch epochs per horizon (K = 4–10)
- Gradient norm clipping
- Separate actor and critic heads with shared CNN backbone

---

## 5. DQN vs PPO — Algorithmic Comparison

This challenge directly compares PPO (Challenge 3) against DQN (Challenge 1) on the same environment under an identical computational budget.

### Key algorithmic differences

| Property | DQN (Challenge 1) | PPO (Challenge 3) |
|---|---|---|
| Learning type | Off-policy | On-policy |
| Experience reuse | ✅ Replay buffer | ❌ Discards after update |
| Exploration | Epsilon-greedy (ε decays) | Entropy bonus |
| Policy update | Q-value targets | Clipped surrogate objective |
| Stability mechanism | Target network | Clip ratio ε |
| Sample efficiency | Higher (reuses data) | Lower (fresh rollouts only) |
| Implementation | Stable-Baselines3 | PyTorch (from scratch) |

### Fair comparison protocol

To ensure results are comparable between both algorithms:

1. **Budget parity** — both agents trained for the same number of environment steps
2. **Identical preprocessing** — grayscale, resize 84×84, frame-stack 4, frame-skip 4
3. **Same seeds** — experiments repeated with seeds 42, 7, and 99
4. **Same environment** — `ALE/Phoenix-v5` with identical ALE version

### Metrics reported

- **Learning curve** — episode return vs. environment steps
- **Sample efficiency** — steps needed to surpass a target score threshold
- **Final performance** — mean ± std over 3 seeds at end of training
- **Training stability** — variance across seeds (area under the curve)
- **Wall-clock time** — total training time on equivalent hardware

### Expected behaviour for Phoenix-v5

Phoenix features fast-moving objects requiring reactive policies. Key observations to investigate:

- PPO with **short horizons (512)** and many epochs suits reactive tasks
- DQN's replay buffer gives it an advantage when environment steps are limited
- Longer horizons (1024, 2048) in PPO may improve credit assignment across multi-wave sequences

---

## 6. Running Experiments

### 6.1 PPO Hyperparameter Sweep

```bash
python challenge3.py --mode sweep --sweep-file sweep_configs_ppo.json
```

### 6.2 DQN Hyperparameter Sweep (Challenge 1)

```bash
python phoenix.py --mode sweep --sweep-file sweep_configs.json
```

### 6.3 Training a Single PPO Model

```bash
python challenge3.py --mode train \
  --model-path models/phoenix_ppo \
  --timesteps 1000000 \
  --seed 42 \
  --tensorboard-log logs/phoenix_ppo
```

### 6.4 Playing a Trained PPO Model

```bash
python challenge3.py --mode play \
  --model-path models/_sweep_ppo_tmp/exp_01_s42 \
  --episodes 5
```

### 6.5 Playing a Trained DQN Model

```bash
python phoenix.py --mode play \
  --model-path models/_sweep_tmp/exp_01_s42 \
  --episodes 5
```

### 6.6 Inspecting a Saved Model

```bash
# PPO
python challenge3.py --mode inspect --model-path models/_sweep_ppo_tmp/exp_01_s42

# DQN
python phoenix.py --mode inspect --model-path models/_sweep_tmp/exp_01_s42
```

---

## 7. CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | *(required)* | `train`, `play`, `inspect`, `sweep` |
| `--sweep-file` | `sweep_configs_ppo.json` | Path to sweep config JSON |
| `--model-path` | `models/phoenix_ppo` | Path to save/load model |
| `--timesteps` | 5,000,000 | Training steps |
| `--episodes` | 3 | Episodes in play mode |
| `--seed` | 42 | Random seed |
| `--tensorboard-log` | `logs/phoenix_ppo` | TensorBoard log directory |

---

## 8. PPO Hyperparameter Search Space

Three configurations were evaluated, each repeated across 3 seeds (42, 7, 99):

| Parameter | exp_01 | exp_02 | exp_03 |
|---|---|---|---|
| `learning_rate` | 1e-4 | 5e-5 | 3e-4 |
| `horizon` | 512 | 256 | 1024 |
| `n_epochs` | 10 | 8 | 12 |
| `batch_size` | 64 | 32 | 64 |
| `gamma` | 0.99 | 0.995 | 0.95 |
| `gae_lambda` | 0.95 | 0.90 | 0.98 |
| `clip_eps` | 0.2 | 0.1 | 0.3 |
| `ent_coef` | 0.01 | 0.005 | 0.02 |
| `vf_coef` | 0.5 | 0.5 | 0.25 |
| `max_grad_norm` | 0.5 | 0.5 | 1.0 |
| `timesteps` | 300,000 | 300,000 | 300,000 |

---

## 9. TensorBoard Visualization

To compare PPO and DQN logs simultaneously:

```bash
tensorboard --logdir logs/
```

Or separately:

```bash
# PPO only
tensorboard --logdir logs/phoenix_ppo

# DQN only
tensorboard --logdir logs/phoenix_dqn
```

Then open: `http://localhost:6006`

Logged metrics include:
- `training/episode_reward` — reward per episode
- `training/loss_policy` — actor (policy) loss *(PPO only)*
- `training/loss_value` — critic (value) loss *(PPO only)*
- `training/entropy` — policy entropy *(PPO only)*
- `training/epsilon` — exploration rate *(DQN only)*

---

## 10. Preprocessing

Identical in both DQN and PPO to ensure fair comparison:

- Grayscale observation
- Resize to 84×84
- Frame stack of 4
- Frame skip of 4
- Pixel values scaled to [0, 1]
- NoOp max of 30 at episode start

---

## 11. Reproducibility

All experiments are fully reproducible:

- Fixed random seeds: **42, 7, 99**
- Deterministic hyperparameter configurations in JSON files
- Fixed training budget: **300,000 steps per run**

To reproduce the best PPO run:

```bash
python challenge3.py --mode sweep --sweep-file sweep_configs_ppo.json --timesteps 300000
```

To reproduce the best DQN run:

```bash
python phoenix.py --mode sweep --sweep-file sweep_configs.json --timesteps 300000
```

---

## 12. Notes

- Training runs on **CPU** by default; a CUDA-capable GPU will significantly speed up training.
- With CPU, each 300k-step run takes approximately **1.5–2 hours**.
- For better performance, increase `--timesteps` to 1M–2M and reduce seeds to 1.
- It is recommended to keep the system connected to power during training.