"""
train_gail.py  —  Challenge 4 / Group 8 / ALE/Phoenix-v5
=========================================================

VERSIÓN MEJORADA:
- Compatible con el nuevo BC
- Inicialización desde BC
- Reward correcta de GAIL
- Label smoothing
- BCEWithLogitsLoss
- Dropout igual al BC
- disc_updates=1 (más estable)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ENV_ID  = "ALE/Phoenix-v5"
N_STACK = 4


# ─────────────────────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_env(seed: int = 0, render_mode=None):
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

    env = gym.make(
        ENV_ID,
        render_mode=render_mode,
        frameskip=1
    )

    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=True,
        grayscale_newaxis=True
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


def compute_gae(
    rewards,
    values,
    dones,
    next_value,
    gamma,
    gae_lambda
):
    advantages = []
    last_gae = 0.0

    vals = [v.item() for v in values] + [next_value]

    for t in reversed(range(len(rewards))):

        mask = 1.0 - float(dones[t])

        delta = (
            rewards[t]
            + gamma * vals[t + 1] * mask
            - vals[t]
        )

        last_gae = (
            delta
            + gamma * gae_lambda * mask * last_gae
        )

        advantages.insert(0, last_gae)

    adv_t = torch.tensor(advantages, dtype=torch.float32)

    ret_t = adv_t + torch.tensor(
        [v.item() for v in values],
        dtype=torch.float32
    )

    return adv_t, ret_t


# ─────────────────────────────────────────────────────────────
# Redes
# ─────────────────────────────────────────────────────────────

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
            nn.Linear(512, n_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        f = self.cnn(x)

        return (
            self.actor(f),
            self.critic(f).squeeze(-1)
        )


class GAILDiscriminator(nn.Module):

    def __init__(
        self,
        n_actions: int,
        use_action: bool = False
    ):
        super().__init__()

        self.use_action = use_action
        self.n_actions = n_actions

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),

            nn.Flatten(),
        )

        fc_in = (
            3136 + n_actions
            if use_action
            else 3136
        )

        self.fc = nn.Sequential(
            nn.Linear(fc_in, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor | None = None
    ) -> torch.Tensor:

        feats = self.cnn(obs)

        if self.use_action and actions is not None:

            acts = actions.long().view(-1)

            one_hot = F.one_hot(
                acts,
                num_classes=self.n_actions
            ).float()

            one_hot = one_hot.view(
                acts.shape[0],
                self.n_actions
            )
            feats = torch.cat(
                [feats, one_hot],
                dim=1
            )

        return self.fc(feats).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# Train GAIL
# ─────────────────────────────────────────────────────────────

def train_gail(
    demos_path: str,
    model_path: str,
    total_steps: int,
    seed: int,
    tensorboard_log: str,

    use_action: bool = False,

    horizon: int = 512,
    n_ppo_epochs: int = 10,
    batch_size: int = 128,

    lr_policy: float = 2.5e-4,
    lr_disc: float = 3e-4,

    disc_updates: int = 1,

    gamma: float = 0.99,
    gae_lambda: float = 0.95,

    clip_eps: float = 0.2,

    ent_coef: float = 0.01,
    vf_coef: float = 0.5,

    max_grad_norm: float = 0.5,

    device_str: str = "cpu",

    bc_init_path: str = "",
):

    set_seed(seed)

    Path(model_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device(device_str)

    print(
        f"GAIL | device={device}"
        f" | use_action={use_action}"
        f" | demos={demos_path}"
        f" | steps={total_steps}"
    )

    # ─────────────────────────────────────────────────────────
    # Demos
    # ─────────────────────────────────────────────────────────

    data = np.load(demos_path)

    demo_obs_np = data["observations"]

    print("\nDemo stats:")
    print("shape:", demo_obs_np.shape)
    print("dtype:", demo_obs_np.dtype)
    print("min:", demo_obs_np.min())
    print("max:", demo_obs_np.max())

    if demo_obs_np.max() > 1.0:
        print("[INFO] Normalizando demos a [0,1]")
        demo_obs_np = demo_obs_np.astype(np.float32) / 255.0

    if demo_obs_np.shape[-1] == 4:
        demo_obs_np = demo_obs_np.transpose(0, 3, 1, 2)

    demo_obs = torch.tensor(
        demo_obs_np,
        dtype=torch.float32
    )

    demo_act = torch.tensor(
        data["actions"],
        dtype=torch.long
    )

    n_demos = len(demo_obs)

    print(
        f"Demostraciones cargadas:"
        f" {n_demos}"
    )

    # ─────────────────────────────────────────────────────────
    # Env y redes
    # ─────────────────────────────────────────────────────────

    env = make_env(seed=seed)

    n_actions = env.action_space.n

    policy = AtariActorCritic(n_actions).to(device)

    disc = GAILDiscriminator(
        n_actions,
        use_action=use_action
    ).to(device)

    # ─────────────────────────────────────────────────────────
    # BC init
    # ─────────────────────────────────────────────────────────

    if bc_init_path and os.path.exists(f"{bc_init_path}.pt"):

        ckpt = torch.load(
            f"{bc_init_path}.pt",
            map_location=device,
            weights_only=False
        )

        policy.load_state_dict(
            ckpt["model_state_dict"]
        )
        
        # Verificación de carga BC
        with torch.no_grad():

            sample_weight = (
                policy.actor[3].weight.mean().item()
            )

        print(
            f"✓ Inicializado desde BC:"
            f" {bc_init_path}.pt"
        )

        print(
            f"[BC CHECK] actor final layer mean weight:"
            f" {sample_weight:.6f}"
        )

    # ─────────────────────────────────────────────────────────
    # Optimizadores
    # ─────────────────────────────────────────────────────────

    opt_policy = optim.Adam(
        policy.parameters(),
        lr=lr_policy
    )

    opt_disc = optim.Adam(
        disc.parameters(),
        lr=lr_disc
    )

    bce = nn.BCEWithLogitsLoss()

    writer = SummaryWriter(
        log_dir=tensorboard_log
    )

    # ─────────────────────────────────────────────────────────
    # Loop
    # ─────────────────────────────────────────────────────────

    obs, _ = env.reset()

    ep_return = 0.0

    all_returns = []

    global_step = 0

    pbar = tqdm(
        total=total_steps,
        desc=f"GAIL seed={seed}",
        unit="step",
        dynamic_ncols=True
    )

    while global_step < total_steps:

        obs_buf = []
        act_buf = []
        logp_buf = []

        rew_real_buf = []
        done_buf = []
        val_buf = []

        # ─────────────────────────────────────────────────────
        # Rollout
        # ─────────────────────────────────────────────────────

        for _ in range(horizon):

            obs_t = obs_to_tensor(obs, device)

            with torch.no_grad():
                logits, value = policy(obs_t)

            dist = Categorical(logits=logits)

            action = dist.sample()

            obs_buf.append(obs_t.squeeze(0).cpu())
            act_buf.append(action.cpu())

            logp_buf.append(
                dist.log_prob(action).cpu()
            )

            val_buf.append(
                value.squeeze().cpu()
            )

            obs, env_reward, terminated, truncated, _ = env.step(
                action.item()
            )

            done = terminated or truncated

            rew_real_buf.append(float(env_reward))
            done_buf.append(done)

            ep_return += float(env_reward)

            global_step += 1

            pbar.update(1)

            if done:

                all_returns.append(ep_return)

                writer.add_scalar(
                    "gail/episode_reward_real",
                    ep_return,
                    global_step
                )

                ep_return = 0.0

                obs, _ = env.reset()

            if global_step >= total_steps:
                break

        obs_stack = torch.stack(obs_buf).to(device)

        act_stack = torch.stack(act_buf).to(device)

        # ─────────────────────────────────────────────────────
        # Reward GAIL CORRECTA
        # ─────────────────────────────────────────────────────

        with torch.no_grad():

            if use_action:
                d_scores = torch.sigmoid(
                    disc(obs_stack, act_stack)
                )
            else:
                d_scores = torch.sigmoid(
                    disc(obs_stack)
                )

            adv_only = (
                -torch.log(1 - d_scores + 1e-8)
            ).cpu().tolist()

            adv_rewards = adv_only

            

        # ─────────────────────────────────────────────────────
        # GAE
        # ─────────────────────────────────────────────────────

        with torch.no_grad():
            _, next_val = policy(
                obs_to_tensor(obs, device)
            )

        advantages, returns = compute_gae(
            adv_rewards,
            val_buf,
            done_buf,
            next_val.item(),
            gamma,
            gae_lambda
        )

        # ─────────────────────────────────────────────────────
        # Discriminador
        # ─────────────────────────────────────────────────────

        actual_horizon = len(obs_buf)

        disc_loss_total = 0.0
        disc_acc_total = 0.0

        for _ in range(disc_updates):

            idx_e = torch.randint(
                0,
                n_demos,
                (batch_size,)
            )

            idx_a = torch.randint(
                0,
                actual_horizon,
                (batch_size,)
            )

            e_obs = demo_obs[idx_e].to(device)
            e_act = demo_act[idx_e].to(device)

            a_obs = obs_stack[idx_a]
            a_act = act_stack[idx_a]

            if use_action:

                d_expert = disc(e_obs, e_act)

                d_agent = disc(a_obs, a_act)

            else:

                d_expert = disc(e_obs)

                d_agent = disc(a_obs)

            real_labels = torch.full_like(
                d_expert,
                0.9
            )

            fake_labels = torch.full_like(
                d_agent,
                0.1
            )

            loss_disc = (
                bce(d_expert, real_labels)
                +
                bce(d_agent, fake_labels)
            )

            opt_disc.zero_grad()

            loss_disc.backward()

            opt_disc.step()

            disc_loss_total += loss_disc.item()

            d_expert_sig = torch.sigmoid(d_expert)
            d_agent_sig = torch.sigmoid(d_agent)

            acc = (
                (
                    d_expert_sig > 0.5
                ).float().mean()
                +
                (
                    d_agent_sig < 0.5
                ).float().mean()
            ) / 2

            disc_acc_total += acc.item()

        avg_disc_loss = (
            disc_loss_total / disc_updates
        )

        avg_disc_acc = (
            disc_acc_total / disc_updates
        )

        writer.add_scalar(
            "gail/disc_loss",
            avg_disc_loss,
            global_step
        )

        writer.add_scalar(
            "gail/disc_accuracy",
            avg_disc_acc,
            global_step
        )

        writer.add_scalar(
            "gail/adv_reward_mean",
            np.mean(adv_rewards),
            global_step
        )
        
        # ─────────────────────────────────────────────────────
        # PPO
        # ─────────────────────────────────────────────────────

        logp_t_all = torch.stack(
            logp_buf
        ).to(device).detach()

        adv_t_all = (
            (
                advantages
                - advantages.mean()
            )
            /
            (
                advantages.std()
                + 1e-8
            )
        ).to(device)

        ret_t_all = returns.to(device)

        for _ in range(n_ppo_epochs):

            idx = torch.randperm(actual_horizon)

            for start in range(
                0,
                actual_horizon,
                batch_size
            ):

                mb = idx[start:start + batch_size]

                lg, vn = policy(obs_stack[mb])

                dist_new = Categorical(logits=lg)

                lp_new = dist_new.log_prob(
                    act_stack[mb]
                )

                ent = dist_new.entropy().mean()

                ratio = (
                    lp_new
                    - logp_t_all[mb]
                ).exp()

                s1 = ratio * adv_t_all[mb]

                s2 = (
                    ratio.clamp(
                        1 - clip_eps,
                        1 + clip_eps
                    )
                    * adv_t_all[mb]
                )

                l_pi = -torch.min(
                    s1,
                    s2
                ).mean()

                l_vf = (
                    (
                        vn
                        - ret_t_all[mb]
                    ) ** 2
                ).mean()

                loss = (
                    l_pi
                    + vf_coef * l_vf
                    - ent_coef * ent
                )

                opt_policy.zero_grad()

                loss.backward()

                nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    max_grad_norm
                )

                opt_policy.step()

        mean_ret = (
            float(np.mean(all_returns[-100:]))
            if all_returns
            else 0.0
        )

        pbar.set_postfix(
            ep=len(all_returns),
            ret=f"{mean_ret:.1f}",
            d_loss=f"{avg_disc_loss:.3f}",
            d_acc=f"{avg_disc_acc:.2f}",
        )

    pbar.close()

    # ─────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────

    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "disc_state_dict": disc.state_dict(),
            "n_actions": n_actions,
        },
        f"{model_path}.pt",
    )

    env.close()

    writer.close()

    final_score = (
        float(np.mean(all_returns[-100:]))
        if all_returns
        else 0.0
    )

    print(
        f"\n✓ Modelo guardado:"
        f" {model_path}.pt"
    )

    print(
        f"Score final:"
        f" {final_score:.1f}"
    )

    return final_score


# ─────────────────────────────────────────────────────────────
# Eval
# ─────────────────────────────────────────────────────────────

def evaluate_gail(
    model_path: str,
    episodes: int = 10,
    seed: int = 42,
    device_str: str = "cpu"
):

    device = torch.device(device_str)

    ckpt = torch.load(
        f"{model_path}.pt",
        map_location=device,
        weights_only=False
    )

    n_actions = ckpt["n_actions"]

    policy = AtariActorCritic(n_actions).to(device)

    policy.load_state_dict(
        ckpt["model_state_dict"]
    )

    policy.eval()

    env = make_env(seed=seed)

    returns = []

    obs, _ = env.reset()

    ep_ret = 0.0

    while len(returns) < episodes:

        obs_t = obs_to_tensor(obs, device)

        with torch.no_grad():
            logits, _ = policy(obs_t)

        dist = Categorical(logits=logits)
        action = dist.sample().item()

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        ep_ret += float(reward)

        if terminated or truncated:

            if info.get("lives", 0) == 0:

                returns.append(ep_ret)

                print(
                    f"Episodio {len(returns)}/{episodes}"
                    f" retorno={ep_ret:.1f}"
                )

                ep_ret = 0.0

            obs, _ = env.reset()

    env.close()

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))

    print(
        f"\nGAIL Evaluación:"
        f" {mean_ret:.1f} ± {std_ret:.1f}"
    )

    return mean_ret

# ─────────────────────────────────────────────────────────────
# Play
# ─────────────────────────────────────────────────────────────

def play_gail(
    model_path: str,
    episodes: int = 3,
    device_str: str = "cpu"
):

    device = torch.device(device_str)

    ckpt = torch.load(
        f"{model_path}.pt",
        map_location=device,
        weights_only=False
    )

    n_actions = ckpt["n_actions"]

    policy = AtariActorCritic(n_actions).to(device)

    policy.load_state_dict(
        ckpt["model_state_dict"]
    )

    policy.eval()

    env = make_env(
        seed=0,
        render_mode="human"
    )

    completed = 0

    obs, _ = env.reset()

    ep_ret = 0.0

    while completed < episodes:

        obs_t = obs_to_tensor(obs, device)

        with torch.no_grad():
            logits, _ = policy(obs_t)

        dist = Categorical(logits=logits)
        action = dist.sample().item()

        obs, reward, terminated, truncated, info = env.step(action)

        ep_ret += float(reward)

        if terminated or truncated:

            if info.get("lives", 0) == 0:

                completed += 1

                print(
                    f"Episodio {completed}/{episodes}"
                    f" retorno={ep_ret:.1f}"
                )

                ep_ret = 0.0

            obs, _ = env.reset()

    env.close()

# ─────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────

def run_sweep(
    sweep_path: str,
    base_log_dir: str,
    bc_init_path: str = ""
):

    seeds = [42, 7, 99]

    with open(sweep_path) as f:
        configs = json.load(f)

    results = []

    best_score = -float("inf")
    best_model_path = None
    best_name = None

    sweep_models_dir = "models/_sweep_gail_tmp"

    Path(sweep_models_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    for idx, cfg in enumerate(configs, 1):

        name = cfg.get(
            "name",
            f"gail_exp_{idx:02d}"
        )

        print(f"\n>>> Experimento: {name}")

        seed_scores = []

        for seed in seeds:

            model_path = (
                f"{sweep_models_dir}/{name}_seed{seed}"
            )

            log_dir = (
                f"{base_log_dir}/{name}/seed_{seed}"
            )

            score = train_gail(
                demos_path      = cfg["demos_path"],
                model_path      = model_path,
                total_steps     = cfg.get("timesteps", 300000),
                seed            = seed,
                tensorboard_log = log_dir,

                use_action      = cfg.get("use_action", False),

                horizon         = cfg.get("horizon", 512),
                n_ppo_epochs    = cfg.get("n_ppo_epochs", 10),
                batch_size      = cfg.get("batch_size", 128),

                lr_policy       = cfg.get("lr_policy", 2.5e-4),
                lr_disc         = cfg.get("lr_disc", 3e-4),

                disc_updates    = cfg.get("disc_updates", 1),

                gamma           = cfg.get("gamma", 0.99),
                gae_lambda      = cfg.get("gae_lambda", 0.95),

                clip_eps        = cfg.get("clip_eps", 0.2),

                ent_coef        = cfg.get("ent_coef", 0.01),
                vf_coef         = cfg.get("vf_coef", 0.5),

                max_grad_norm   = cfg.get("max_grad_norm", 0.5),

                bc_init_path    = bc_init_path,
            )

            seed_scores.append(score)

            # Trackear el mejor modelo individual (experimento + seed)
            if score > best_score:
                best_score = score
                best_model_path = f"{sweep_models_dir}/{name}_seed{seed}.pt"
                best_name = f"{name}_seed{seed}"

        avg_score = float(np.mean(seed_scores))

        results.append(
            (name, avg_score)
        )

        print(
            f"{name} promedio:"
            f" {avg_score:.1f}"
        )

    print("\n=== RESULTADOS SWEEP ===")

    for name, score in sorted(
        results,
        key=lambda x: -x[1]
    ):
        print(f"{name}: {score:.1f}")

    if best_model_path is not None:

        shutil.copy(
            best_model_path,
            "models/best_gail.pt"
        )

        print(
            f"\n✓ Mejor modelo guardado como:"
            f" models/best_gail.pt"
        )

        print(
            f"Mejor experimento:"
            f" {best_name}"
            f" | score={best_score:.1f}"
        )

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():

    p = argparse.ArgumentParser(
        description="GAIL — Challenge 4"
    )

    p.add_argument(
        "--mode",
        choices=["train", "eval", "play", "sweep"],
        default="train"
    )

    p.add_argument(
        "--demos",
        default="demos/demos_dqn_20k.npz"
    )

    p.add_argument(
        "--model-path",
        default="models/phoenix_gail"
    )

    p.add_argument(
        "--sweep-file",
        default="sweep_configs_gail.json"
    )

    p.add_argument(
        "--timesteps",
        type=int,
        default=300_000
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42
    )

    p.add_argument(
        "--use-action",
        type=lambda x: x.lower() == "true",
        default=False
    )

    p.add_argument(
        "--episodes",
        type=int,
        default=10
    )

    p.add_argument(
        "--tensorboard-log",
        default="logs/phoenix_gail"
    )

    p.add_argument(
        "--device",
        default="cpu"
    )

    p.add_argument(
        "--bc-init",
        default="models/phoenix_bc"
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train_gail(
            demos_path      = args.demos,
            model_path      = args.model_path,
            total_steps     = args.timesteps,
            seed            = args.seed,
            tensorboard_log = args.tensorboard_log,
            use_action      = args.use_action,
            device_str      = args.device,
            bc_init_path    = args.bc_init,
        )

    elif args.mode == "play":
        play_gail(
            model_path = args.model_path,
            episodes   = args.episodes,
            device_str = args.device,
        )

    elif args.mode == "eval":
        evaluate_gail(
            model_path = args.model_path,
            episodes   = args.episodes,
            seed       = args.seed,
            device_str = args.device,
        )

    else:
        run_sweep(
            sweep_path   = args.sweep_file,
            base_log_dir = args.tensorboard_log,
            bc_init_path = args.bc_init,
        )