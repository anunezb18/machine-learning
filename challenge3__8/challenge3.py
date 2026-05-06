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
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ENV_ID = "ALE/Phoenix-v5"
N_STACK = 4


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def make_env(env_id: str = ENV_ID, seed: int = 0, render_mode: str | None = None):
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

    env = gym.make(env_id, render_mode=render_mode, frameskip=1)
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


def obs_to_tensor(obs, device: torch.device) -> torch.Tensor:
    """
    Convierte la observación de FrameStackObservation a (1, N_STACK, 84, 84).

    grayscale_newaxis=True puede devolver shapes como:
      (84, 84, 1, N_STACK)  o  (84, 84, N_STACK)  o  (N_STACK, 84, 84)
    Esta función normaliza cualquiera de ellos.
    """
    arr = np.array(obs, dtype=np.float32)
    arr = arr.squeeze()   # elimina ejes de tamaño 1

    if arr.ndim == 3:
        if arr.shape[0] == N_STACK:
            pass                        # ya es (N_STACK, 84, 84)
        elif arr.shape[2] == N_STACK:
            arr = arr.transpose(2, 0, 1)  # (84,84,N) -> (N,84,84)
    elif arr.ndim == 2:
        arr = arr[np.newaxis, ...]      # edge-case frame único

    return torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, N_STACK, 84, 84)


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------

class AtariActorCritic(nn.Module):

    def __init__(self, n_actions: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out = 64 * 7 * 7

        self.actor = nn.Sequential(
            nn.Linear(cnn_out, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(cnn_out, 512), nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor):
        feats = self.cnn(x)
        return self.actor(feats), self.critic(feats).squeeze(-1)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    advantages = []
    last_gae = 0.0
    values_np = [v.item() for v in values] + [next_value]

    for t in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values_np[t + 1] * mask - values_np[t]
        last_gae = delta + gamma * gae_lambda * mask * last_gae
        advantages.insert(0, last_gae)

    advantages_t = torch.tensor(advantages, dtype=torch.float32)
    returns_t = advantages_t + torch.tensor([v.item() for v in values], dtype=torch.float32)
    return advantages_t, returns_t


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ppo(model_path, total_steps, seed, tensorboard_log, hparams=None):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    if hparams is None:
        hparams = dict(
            env_id=ENV_ID, learning_rate=2.5e-4, horizon=512, n_epochs=10,
            batch_size=128, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            total_steps=total_steps, seed=seed,
        )

    writer = SummaryWriter(log_dir=tensorboard_log)
    writer.add_hparams(
        {k: v for k, v in hparams.items() if isinstance(v, (int, float, str, bool))},
        metric_dict={"hparam/episode_reward": 0},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = make_env(ENV_ID, seed=seed)
    n_actions = env.action_space.n
    model = AtariActorCritic(n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hparams["learning_rate"])

    horizon      = hparams["horizon"]
    n_epochs     = hparams["n_epochs"]
    batch_size   = hparams["batch_size"]
    gamma        = hparams["gamma"]
    gae_lambda   = hparams["gae_lambda"]
    clip_eps     = hparams["clip_eps"]
    ent_coef     = hparams["ent_coef"]
    vf_coef      = hparams["vf_coef"]
    max_grad_norm = hparams["max_grad_norm"]

    obs, _ = env.reset()
    episode_return = 0.0
    all_returns: list[float] = []
    global_step = 0

    pbar = tqdm(total=total_steps, desc=f"Seed {seed}", unit="step", dynamic_ncols=True)

    while global_step < total_steps:
        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf, done_buf, val_buf = [], [], []

        for _ in range(horizon):
            obs_t = obs_to_tensor(obs, device)          # (1, 4, 84, 84)

            with torch.no_grad():
                logits, value = model(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()

            obs_buf.append(obs_t.squeeze(0).cpu())      # (4, 84, 84)
            act_buf.append(action.cpu())
            logp_buf.append(dist.log_prob(action).cpu())
            val_buf.append(value.squeeze().cpu())

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            rew_buf.append(float(reward))
            done_buf.append(done)
            episode_return += float(reward)
            global_step += 1

            pbar.update(1)
            pbar.set_postfix(ep=len(all_returns), ret=f"{np.mean(all_returns[-100:]):.1f}" if all_returns else "0.0")

            if done:
                all_returns.append(episode_return)
                writer.add_scalar("training/episode_reward", episode_return, global_step)
                episode_return = 0.0
                obs, _ = env.reset()

            if global_step >= total_steps:
                break

        with torch.no_grad():
            _, next_val = model(obs_to_tensor(obs, device))

        advantages, returns = compute_gae(
            rew_buf, val_buf, done_buf, next_val.item(), gamma, gae_lambda
        )

        actual_horizon = len(obs_buf)
        obs_t_all  = torch.stack(obs_buf).to(device)
        act_t_all  = torch.stack(act_buf).to(device)
        logp_t_all = torch.stack(logp_buf).to(device).detach()
        adv_t_all  = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).to(device)
        ret_t_all  = returns.to(device)

        total_loss_pi = total_loss_vf = total_entropy = 0.0
        update_count = 0

        for _ in range(n_epochs):
            idx = torch.randperm(actual_horizon)
            for start in range(0, actual_horizon, batch_size):
                mb = idx[start: start + batch_size]
                logits, val_new = model(obs_t_all[mb])
                dist_new = Categorical(logits=logits)
                logp_new = dist_new.log_prob(act_t_all[mb])
                entropy  = dist_new.entropy().mean()

                ratio = (logp_new - logp_t_all[mb]).exp()
                surr1 = ratio * adv_t_all[mb]
                surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_t_all[mb]
                loss_pi = -torch.min(surr1, surr2).mean()
                loss_vf = ((val_new - ret_t_all[mb]) ** 2).mean()
                loss    = loss_pi + vf_coef * loss_vf - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                total_loss_pi += loss_pi.item()
                total_loss_vf += loss_vf.item()
                total_entropy += entropy.item()
                update_count  += 1

        if update_count > 0:
            writer.add_scalar("training/loss_policy", total_loss_pi / update_count, global_step)
            writer.add_scalar("training/loss_value",  total_loss_vf / update_count, global_step)
            writer.add_scalar("training/entropy",     total_entropy  / update_count, global_step)

    pbar.close()

    torch.save(
        {"model_state_dict": model.state_dict(), "n_actions": n_actions, "hparams": hparams},
        f"{model_path}.pt",
    )
    env.close()
    writer.close()

    return float(np.mean(all_returns[-100:])) if all_returns else 0.0


# ---------------------------------------------------------------------------
# Play
# ---------------------------------------------------------------------------

def play_agent(model_path: str, episodes: int) -> None:
    pt_path = f"{model_path}.pt"
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Model not found: {pt_path}")

    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
    model = AtariActorCritic(checkpoint["n_actions"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device("cpu")
    env = make_env(ENV_ID, seed=0, render_mode="human")

    completed = 0
    obs, _ = env.reset()
    episode_reward = 0.0

    while completed < episodes:
        with torch.no_grad():
            logits, _ = model(obs_to_tensor(obs, device))
        action = logits.argmax(dim=-1).item()

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += float(reward)

        if terminated or truncated:
            if info.get("lives", 0) == 0:
                completed += 1
                print(f"Episode {completed}/{episodes} reward: {episode_reward:.2f}")
                episode_reward = 0.0
            obs, _ = env.reset()

    env.close()


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(sweep_path: str, default_timesteps: int, base_log_dir: str) -> None:
    seeds = [42, 7, 99]

    with open(sweep_path) as f:
        configs = json.load(f)

    tmp_model_dir = Path("models") / "_sweep_ppo_tmp"
    tmp_model_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, float]] = []

    for idx, cfg in enumerate(configs, start=1):
        name = cfg.get("name", f"exp_{idx:02d}")
        exp_timesteps = cfg.get("timesteps", default_timesteps)
        seed_scores: list[float] = []
        print(f"\n>>> Iniciando Experimento PPO: {name} ({exp_timesteps} steps)")

        for current_seed in seeds:
            print(f"  --- Ejecutando Semilla: {current_seed} ---")
            hparams = {
                "env_id": ENV_ID,
                "learning_rate":        cfg["learning_rate"],
                "horizon":              cfg["horizon"],
                "n_epochs":             cfg["n_epochs"],
                "batch_size":           cfg["batch_size"],
                "gamma":                cfg["gamma"],
                "gae_lambda":           cfg["gae_lambda"],
                "clip_eps":             cfg["clip_eps"],
                "ent_coef":             cfg["ent_coef"],
                "vf_coef":              cfg["vf_coef"],
                "max_grad_norm":        cfg["max_grad_norm"],
                "total_steps":          exp_timesteps,
                "seed":                 current_seed,
            }
            log_dir    = f"{base_log_dir}/sweep/{name}/seed_{current_seed}"
            model_path = str(tmp_model_dir / f"{name}_s{current_seed}")
            score = train_ppo(model_path=model_path, total_steps=exp_timesteps,
                              seed=current_seed, tensorboard_log=log_dir, hparams=hparams)
            seed_scores.append(score)

        avg_score = float(np.mean(seed_scores))
        results.append((name, avg_score))
        print(f"  => Recompensa promedio del experimento {name}: {avg_score:.2f}")

    print("\n=== Resultados del Sweep PPO ===")
    for name, score in sorted(results, key=lambda x: -x[1]):
        print(f"  {name}: {score:.2f}")


# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------

def inspect_model(model_path: str) -> None:
    pt_path = f"{model_path}.pt"
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Model not found: {pt_path}")
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
    print(f"n_actions: {checkpoint['n_actions']}")
    for key, value in checkpoint.get("hparams", {}).items():
        print(f"{key}: {value}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO agent for ALE/Phoenix-v5")
    parser.add_argument("--mode", choices=["train", "play", "inspect", "sweep"], required=True)
    parser.add_argument("--sweep-file",      default="sweep_configs_ppo.json")
    parser.add_argument("--model-path",      default="models/phoenix_ppo")
    parser.add_argument("--timesteps",       type=int, default=None)
    parser.add_argument("--episodes",        type=int, default=3)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--tensorboard-log", default="logs/phoenix_ppo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "train":
        train_ppo(
            model_path=args.model_path,
            total_steps=args.timesteps or 5_000_000,
            seed=args.seed,
            tensorboard_log=args.tensorboard_log,
        )
    elif args.mode == "play":
        play_agent(model_path=args.model_path, episodes=args.episodes)
    elif args.mode == "sweep":
        run_sweep(
            sweep_path=args.sweep_file,
            default_timesteps=args.timesteps or 5_000_000,
            base_log_dir=args.tensorboard_log,
        )
    else:
        inspect_model(model_path=args.model_path)


if __name__ == "__main__":
    main()