"""
collect_demos.py  —  Challenge 4 / Group 8 / ALE/Phoenix-v5
============================================================
Graba pares (observación, acción) desde un checkpoint ya entrenado
(DQN .zip de SB3 o PPO .pt propio) y los guarda en un archivo .npz.

Uso:
    python collect_demos.py --checkpoint models/phoenix_best.zip \
                            --source dqn --n-steps 20000 --seed 42 \
                            --out demos/demos_dqn_20k.npz

    python collect_demos.py --checkpoint models/phoenix_best.zip \
                            --source dqn --n-steps 5000 --seed 42 \
                            --out demos/demos_dqn_5k.npz
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

import torch
import torch.nn as nn

ENV_ID  = "ALE/Phoenix-v5"
N_STACK = 4


# ── Red Actor-Critic (para checkpoints PPO propios) ────────────────────────

class AtariActorCritic(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        self.actor  = nn.Sequential(nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, n_actions))
        self.critic = nn.Sequential(nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, x):
        f = self.cnn(x)
        return self.actor(f), self.critic(f).squeeze(-1)


# ── Cargador DQN — mismo entorno que phoenix.py del Challenge 1 ────────────

def load_dqn(checkpoint_path: str, device):
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import VecFrameStack

    vec_env = make_atari_env(ENV_ID, n_envs=1, seed=0)
    vec_env = VecFrameStack(vec_env, n_stack=N_STACK)
    model   = DQN.load(checkpoint_path, env=vec_env, device=device)
    return model, vec_env


# ── Cargador PPO propio ────────────────────────────────────────────────────

def make_env(seed: int = 0):
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
    env = gym.make(ENV_ID, frameskip=1)
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84,
                             grayscale_obs=True, scale_obs=True,
                             grayscale_newaxis=True)
    env = FrameStackObservation(env, N_STACK)
    env.reset(seed=seed)
    return env

def obs_to_tensor(obs, device):
    arr = np.array(obs, dtype=np.float32).squeeze()
    if arr.ndim == 3 and arr.shape[0] != N_STACK:
        arr = arr.transpose(2, 0, 1)
    elif arr.ndim == 2:
        arr = arr[np.newaxis]
    return torch.from_numpy(arr).unsqueeze(0).to(device)

def load_ppo_policy(checkpoint_path: str, n_actions: int, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net  = AtariActorCritic(n_actions).to(device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    def predict(obs_t: torch.Tensor) -> int:
        with torch.no_grad():
            logits, _ = net(obs_t.to(device))
        return int(logits.argmax(dim=-1).item())

    return predict


# ── Colección principal ────────────────────────────────────────────────────

def collect_demonstrations(
    checkpoint_path: str,
    source: str,
    n_steps: int,
    seed: int,
    out_path: str,
    device_str: str = "cpu",
) -> dict:
    device = torch.device(device_str)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    obs_buf, act_buf = [], []
    episode_returns, episode_return = [], 0.0

    if source == "dqn":
        model, vec_env = load_dqn(checkpoint_path, device)
        obs = vec_env.reset()  # (1, 4, 84, 84) uint8

        print(f"Recolectando {n_steps} pasos desde DQN...")
        for step in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(action[0])

            obs_buf.append((obs[0].astype(np.float32) / 255.0))  # (4,84,84) float32
            act_buf.append(action_int)

            obs, rewards, dones, infos = vec_env.step(action)
            episode_return += float(rewards[0])

            if dones[0]:
                episode_returns.append(episode_return)
                episode_return = 0.0

            if (step + 1) % 5000 == 0:
                avg = np.mean(episode_returns[-20:]) if episode_returns else 0.0
                print(f"  {step+1}/{n_steps} pasos  |  episodios={len(episode_returns)}"
                      f"  |  ret_media(20)={avg:.1f}")

        vec_env.close()

    elif source == "ppo":
        env = make_env(seed=seed)
        n_actions = env.action_space.n
        predict = load_ppo_policy(checkpoint_path, n_actions, device)
        obs, _ = env.reset()

        print(f"Recolectando {n_steps} pasos desde PPO...")
        for step in range(n_steps):
            obs_t  = obs_to_tensor(obs, device)
            action = predict(obs_t)

            obs_buf.append(obs_t.squeeze(0).cpu().numpy())
            act_buf.append(action)

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)

            if terminated or truncated:
                episode_returns.append(episode_return)
                episode_return = 0.0
                obs, _ = env.reset()

            if (step + 1) % 5000 == 0:
                avg = np.mean(episode_returns[-20:]) if episode_returns else 0.0
                print(f"  {step+1}/{n_steps} pasos  |  episodios={len(episode_returns)}"
                      f"  |  ret_media(20)={avg:.1f}")

        env.close()

    else:
        raise ValueError(f"source debe ser 'dqn' o 'ppo', recibido: {source}")

    demos = {
        "observations": np.array(obs_buf, dtype=np.float32),
        "actions":      np.array(act_buf, dtype=np.int64),
    }
    np.savez_compressed(out_path, **demos)

    meta = {
        "source":      source,
        "n_steps":     n_steps,
        "n_episodes":  len(episode_returns),
        "seed":        seed,
        "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "std_return":  float(np.std(episode_returns))  if episode_returns else 0.0,
        "min_return":  float(np.min(episode_returns))  if episode_returns else 0.0,
        "max_return":  float(np.max(episode_returns))  if episode_returns else 0.0,
    }
    meta_path = out_path.replace(".npz", "_info.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Demos guardadas en:  {out_path}")
    print(f"  Pasos:     {n_steps}")
    print(f"  Episodios: {len(episode_returns)}")
    if episode_returns:
        print(f"  Retorno del demostrador: {meta['mean_return']:.1f} ± {meta['std_return']:.1f}")

    unique, counts = np.unique(demos["actions"], return_counts=True)
    print(f"  Distribución de acciones: { {int(u): int(c) for u, c in zip(unique, counts)} }")

    return demos


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Recolector de demostraciones — Challenge 4")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source",     required=True, choices=["dqn", "ppo"])
    p.add_argument("--n-steps",    type=int, default=20_000)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--out",        default="demos/demos.npz")
    p.add_argument("--device",     default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_demonstrations(
        checkpoint_path=args.checkpoint,
        source=args.source,
        n_steps=args.n_steps,
        seed=args.seed,
        out_path=args.out,
        device_str=args.device,
    )