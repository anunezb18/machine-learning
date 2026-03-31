from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

ENV_ID = "ALE/Phoenix-v5"
N_STACK = 4


class TensorBoardCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self._writer: SummaryWriter | None = None
        self._episode_reward = 0.0

    def _on_training_start(self) -> None:
        from stable_baselines3.common.logger import TensorBoardOutputFormat
        for fmt in self.model._logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                self._writer = fmt.writer
                return
        self._writer = None

    def _on_step(self) -> bool:
        if self._writer is None:
            return True

        self._episode_reward += float(self.locals["rewards"][0])

        self._writer.add_scalar(
            "training/epsilon",
            self.model.exploration_rate,
            self.num_timesteps,
        )

        if self.locals["dones"][0]:
            self._writer.add_scalar(
                "training/episode_reward",
                self._episode_reward,
                self.num_timesteps,
            )
            self._episode_reward = 0.0

        return True


def build_training_environment(seed: int) -> VecFrameStack:
    env = make_atari_env(ENV_ID, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=N_STACK)
    return env


def build_playing_environment() -> VecFrameStack:
    def _make_single_env() -> AtariWrapper:
        base_env = gym.make(ENV_ID, render_mode="human")
        return AtariWrapper(base_env, terminal_on_life_loss=True, clip_reward=False)

    env = DummyVecEnv([_make_single_env])
    env = VecFrameStack(env, n_stack=N_STACK)
    return env


def train_agent(
    model_path: str,
    timesteps: int,
    seed: int,
    tensorboard_log: str,
    hparams: dict | None = None,
) -> float:
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    if hparams is None:
        hparams = dict(
            env_id=ENV_ID,
            learning_rate=1e-4,
            buffer_size=50_000,
            learning_starts=10_000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1_000,
            exploration_fraction=0.25,
            exploration_final_eps=0.01,
            timesteps=timesteps,
            seed=seed,
        )

    _tb_writer = SummaryWriter(log_dir=tensorboard_log)
    _tb_writer.add_hparams(hparams, metric_dict={"hparam/episode_reward": 0})
    _tb_writer.close()

    env = build_training_environment(seed=seed)

    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=hparams["learning_rate"],
        buffer_size=hparams["buffer_size"],
        learning_starts=hparams["learning_starts"],
        batch_size=hparams["batch_size"],
        tau=1.0,
        gamma=hparams["gamma"],
        train_freq=hparams["train_freq"],
        gradient_steps=1,
        target_update_interval=hparams["target_update_interval"],
        exploration_fraction=hparams["exploration_fraction"],
        exploration_final_eps=hparams["exploration_final_eps"],
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=seed,
    )

    model.learn(
        total_timesteps=timesteps,
        callback=TensorBoardCallback(),
        progress_bar=True,
    )
    model.save(model_path)
    env.close()

    if model.ep_info_buffer:
        return float(np.mean([ep["r"] for ep in model.ep_info_buffer]))
    return 0.0


def play_agent(model_path: str, episodes: int) -> None:
    if not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(f"Model not found: {model_path}.zip")

    env = build_playing_environment()
    model = DQN.load(model_path, env=env)

    completed = 0
    obs = env.reset()
    episode_reward = 0.0

    while completed < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        episode_reward += float(rewards[0])

        if dones[0]:
            if infos[0].get("lives", 0) == 0:
                completed += 1
                print(f"Episode {completed}/{episodes} reward: {episode_reward:.2f}")
                episode_reward = 0.0

    env.close()


def run_sweep(
    sweep_path: str,
    default_timesteps: int,
    base_log_dir: str,
    best_model_path: str,
) -> None:

    seeds = [42, 7, 99]

    with open(sweep_path) as f:
        configs = json.load(f)

    tmp_model_dir = Path("models") / "_sweep_tmp"
    tmp_model_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, float]] = []

    for idx, cfg in enumerate(configs, start=1):
        name = cfg.get("name", f"exp_{idx:02d}")
        exp_timesteps = cfg.get("timesteps", default_timesteps)

        seed_scores = []
        print(f"\n>>> Iniciando Experimento: {name} ({exp_timesteps} steps)")

        for current_seed in seeds:
            print(f"  --- Ejecutando Semilla: {current_seed} ---")

            hparams = {
                "env_id": ENV_ID,
                "learning_rate": cfg["learning_rate"],
                "buffer_size": cfg["buffer_size"],
                "learning_starts": cfg["learning_starts"],
                "batch_size": cfg["batch_size"],
                "gamma": cfg["gamma"],
                "train_freq": cfg["train_freq"],
                "target_update_interval": cfg["target_update_interval"],
                "exploration_fraction": cfg["exploration_fraction"],
                "exploration_final_eps": cfg["exploration_final_eps"],
                "timesteps": exp_timesteps,
                "seed": current_seed,
            }

            log_dir = f"{base_log_dir}/sweep/{name}/seed_{current_seed}"
            model_path = str(tmp_model_dir / f"{name}_s{current_seed}")

            score = train_agent(
                model_path=model_path,
                timesteps=exp_timesteps,
                seed=current_seed,
                tensorboard_log=log_dir,
                hparams=hparams,
            )

            seed_scores.append(score)

        avg_score = float(np.mean(seed_scores))
        results.append((name, avg_score))

        print(f"  => Recompensa promedio del experimento {name}: {avg_score:.2f}")


def inspect_model(model_path: str) -> None:
    if not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(f"Model not found: {model_path}.zip")

    model = DQN.load(model_path)

    params = {
        "policy": model.policy_class.__name__,
        "learning_rate": model.learning_rate,
        "buffer_size": model.buffer_size,
        "learning_starts": model.learning_starts,
        "batch_size": model.batch_size,
        "gamma": model.gamma,
        "train_freq": model.train_freq,
        "target_update_interval": model.target_update_interval,
        "exploration_fraction": model.exploration_fraction,
        "exploration_final_eps": model.exploration_final_eps,
        "num_timesteps_trained": model.num_timesteps,
    }

    for key, value in params.items():
        print(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "play", "inspect", "sweep"], required=True)
    parser.add_argument("--sweep-file", default="sweep_configs.json")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--model-path", default="models/phoenix_dqn")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensorboard-log", default="logs/phoenix_dqn")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "train":
        timesteps = args.timesteps or 500_000

        train_agent(
            model_path=args.model_path,
            timesteps=timesteps,
            seed=args.seed,
            tensorboard_log=args.tensorboard_log,
        )

    elif args.mode == "play":
        play_agent(model_path=args.model_path, episodes=args.episodes)

    elif args.mode == "sweep":
        run_sweep(
            sweep_path=args.sweep_file,
            default_timesteps=args.timesteps or 500_000,
            base_log_dir=args.tensorboard_log,
            best_model_path=args.model_path,
        )

    else:
        inspect_model(model_path=args.model_path)


if __name__ == "__main__":
    main()