"""
train_bc.py  —  Challenge 4 / Group 8 / ALE/Phoenix-v5
=======================================================
Behavioral Cloning mejorado para Atari Phoenix.

Mejoras implementadas:
- Verificación y normalización automática del dataset
- Más epochs por defecto
- Weight decay
- Dropout
- Data augmentation ligera
- Evaluación estocástica (sampling)
- Seeds reproducibles
- Guardado por mejor evaluación real, no por loss
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

ENV_ID  = "ALE/Phoenix-v5"
N_STACK = 4


# ── Reproducibilidad ───────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Environment ────────────────────────────────────────────────────────────

def make_env(seed: int = 0, render_mode=None):
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

    env = gym.make(ENV_ID, render_mode=render_mode, frameskip=1)

    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=True,
        grayscale_newaxis=True,
    )

    env = FrameStackObservation(env, N_STACK)

    env.reset(seed=seed)
    return env


def obs_to_tensor(obs, device):
    arr = np.array(obs, dtype=np.float32).squeeze()

    if arr.ndim == 3:
        if arr.shape[0] != N_STACK:
            arr = arr.transpose(2, 0, 1)

    elif arr.ndim == 2:
        arr = arr[np.newaxis]

    return torch.from_numpy(arr).unsqueeze(0).to(device)


# ── Red ────────────────────────────────────────────────────────────────────

class AtariActorCritic(nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),

            nn.Flatten(),
        )

        self.actor = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        f = self.cnn(x)
        return self.actor(f), self.critic(f).squeeze(-1)


# ── Evaluación interna ─────────────────────────────────────────────────────

def evaluate_policy(
    model,
    device,
    episodes=3,
    seed=42,
):
    model.eval()

    env = make_env(seed=seed)

    returns = []

    obs, _ = env.reset()
    ep_ret = 0.0

    while len(returns) < episodes:

        obs_t = obs_to_tensor(obs, device)

        with torch.no_grad():
            logits, _ = model(obs_t)

        # Sampling en vez de argmax
        dist = Categorical(logits=logits)
        action = dist.sample().item()

        obs, reward, terminated, truncated, _ = env.step(action)

        ep_ret += float(reward)

        if terminated or truncated:
            returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()

    env.close()

    return float(np.mean(returns))


# ── Entrenamiento BC ───────────────────────────────────────────────────────

def train_bc(
    demos_path: str,
    model_path: str,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-4,
    tensorboard_log: str = "logs/phoenix_bc",
    device_str: str = "cpu",
    seed: int = 42,
):

    set_seed(seed)

    device = torch.device(device_str)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────

    data = np.load(demos_path)

    obs_np = data["observations"]
    act_np = data["actions"]

    if obs_np.shape[-1] == 4:
        obs_np = obs_np.transpose(0, 3, 1, 2)

    print(f"\nDataset cargado:")
    print(f"  shape: {obs_np.shape}")
    print(f"  dtype: {obs_np.dtype}")
    print(f"  min:   {obs_np.min()}")
    print(f"  max:   {obs_np.max()}")
    print(f"  acciones únicas: {np.unique(act_np)}")

    # ── Normalización automática ──────────────────────────────────────────

    if obs_np.max() > 1.0:
        print("\n[INFO] Normalizando observaciones a [0,1]")
        obs_np = obs_np.astype(np.float32) / 255.0

    obs_t = torch.tensor(obs_np, dtype=torch.float32)
    act_t = torch.tensor(act_np, dtype=torch.long)

    dataset = TensorDataset(obs_t, act_t)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # ── Modelo ────────────────────────────────────────────────────────────

    env = make_env(seed=seed)
    n_actions = env.action_space.n
    env.close()

    model = AtariActorCritic(n_actions).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )

    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=tensorboard_log)

    best_eval = -float("inf")

    global_batch = 0

    # ── Entrenamiento ─────────────────────────────────────────────────────

    for epoch in range(1, n_epochs + 1):

        model.train()

        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        for obs_b, act_b in loader:

            obs_b = obs_b.to(device)
            act_b = act_b.to(device)

            # Data augmentation ligera
            noise = 0.01 * torch.randn_like(obs_b)
            obs_b = (obs_b + noise).clamp(0, 1)

            logits, _ = model(obs_b)

            loss = criterion(logits, act_b)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            preds = logits.argmax(dim=-1)

            acc = (preds == act_b).float().mean().item()

            epoch_loss += loss.item()
            epoch_acc += acc
            n_batches += 1
            global_batch += 1

            writer.add_scalar(
                "bc/batch_loss",
                loss.item(),
                global_batch
            )

        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / n_batches

        writer.add_scalar("bc/epoch_loss", avg_loss, epoch)
        writer.add_scalar("bc/epoch_accuracy", avg_acc, epoch)

        print(
            f"Epoch {epoch:>3}/{n_epochs}  "
            f"loss={avg_loss:.4f}  "
            f"acc={avg_acc:.3f}"
        )

        # ── Evaluación periódica ──────────────────────────────────────────

        if epoch % 5 == 0:

            eval_score = evaluate_policy(
                model,
                device,
                episodes=3,
                seed=seed,
            )

            writer.add_scalar(
                "bc/eval_return",
                eval_score,
                epoch
            )

            print(f"  → Eval return: {eval_score:.1f}")

            # Guardar por score real
            if eval_score > best_eval:

                best_eval = eval_score

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "n_actions": n_actions,
                        "hparams": {
                            "lr": lr,
                            "batch_size": batch_size,
                            "n_epochs": n_epochs,
                            "seed": seed,
                            "demos_path": demos_path,
                        },
                    },
                    f"{model_path}.pt",
                )

                print(
                    f"  ✓ Nuevo mejor modelo "
                    f"(score={best_eval:.1f})"
                )

    writer.close()

    print(
        f"\n✓ Modelo BC guardado en "
        f"{model_path}.pt"
    )

    print(f"✓ Mejor evaluación: {best_eval:.1f}")

    return model


# ── Evaluación ─────────────────────────────────────────────────────────────

def evaluate_bc(
    model_path: str,
    episodes: int = 10,
    seed: int = 42,
    device_str: str = "cpu",
):

    set_seed(seed)

    device = torch.device(device_str)

    pt_path = f"{model_path}.pt"

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"No se encontró: {pt_path}")

    ckpt = torch.load(
        pt_path,
        map_location=device,
        weights_only=False,
    )

    n_actions = ckpt["n_actions"]

    model = AtariActorCritic(n_actions).to(device)

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.eval()

    env = make_env(seed=seed)

    returns = []

    obs, _ = env.reset()

    ep_ret = 0.0

    while len(returns) < episodes:

        obs_t = obs_to_tensor(obs, device)

        with torch.no_grad():
            logits, _ = model(obs_t)

        # Sampling en vez de greedy
        dist = Categorical(logits=logits)
        action = dist.sample().item()

        obs, reward, terminated, truncated, _ = env.step(action)

        ep_ret += float(reward)

        if terminated or truncated:

            returns.append(ep_ret)

            print(
                f"  Episodio {len(returns)}/{episodes}  "
                f"retorno={ep_ret:.1f}"
            )

            ep_ret = 0.0

            obs, _ = env.reset()

    env.close()

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))

    print(
        f"\nBC Evaluación ({episodes} episodios): "
        f"{mean_ret:.1f} ± {std_ret:.1f}"
    )

    return mean_ret


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():

    p = argparse.ArgumentParser(
        description="Behavioral Cloning — Phoenix"
    )

    p.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
    )

    p.add_argument(
        "--demos",
        default="demos/demos_dqn_20k.npz",
    )

    p.add_argument(
        "--model-path",
        default="models/phoenix_bc",
    )

    p.add_argument(
        "--n-epochs",
        type=int,
        default=100,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    p.add_argument(
        "--episodes",
        type=int,
        default=10,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--tensorboard-log",
        default="logs/phoenix_bc",
    )

    p.add_argument(
        "--device",
        default="cpu",
    )

    return p.parse_args()


if __name__ == "__main__":

    args = parse_args()

    if args.mode == "train":

        train_bc(
            demos_path=args.demos,
            model_path=args.model_path,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            tensorboard_log=args.tensorboard_log,
            device_str=args.device,
            seed=args.seed,
        )

    else:

        evaluate_bc(
            model_path=args.model_path,
            episodes=args.episodes,
            seed=args.seed,
            device_str=args.device,
        )