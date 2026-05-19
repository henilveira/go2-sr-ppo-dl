"""
Decomposed PPO training for Go2 self-recovery.

Usage:
    python src/models/qdecomp_ppo/train.py
    python src/models/qdecomp_ppo/train.py --timesteps 50000
"""

import sys
import argparse
from pathlib import Path
import yaml
import numpy as np
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from environment.go2_env import Go2Env
from src.models.qdecomp_ppo.agent import DecompPPO
from src.models.qdecomp_ppo.networks import REWARD_KEYS


def load_config():
    with open(project_root / "config" / "train_config.yml") as f:
        return yaml.safe_load(f)


def make_env(config):
    return Go2Env(config, render_mode=None)


def evaluate(agent: DecompPPO, env: Go2Env, n_episodes: int = 5) -> dict:
    total_rewards, successes = [], 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_r, done = 0.0, False
        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_r += reward
            done  = terminated or truncated
        total_rewards.append(ep_r)
        if info.get('reward_breakdown', {}).get('R_g', 0.0) > 0.8:
            successes += 1
    return {
        'eval/mean_reward':  float(np.mean(total_rewards)),
        'eval/std_reward':   float(np.std(total_rewards)),
        'eval/success_rate': successes / n_episodes,
    }


def train(total_timesteps_override: int | None = None):
    print("=" * 70)
    print("Decomposed PPO — Go2 Self-Recovery")
    print(f"  Subagentes: {len(REWARD_KEYS)} ({', '.join(REWARD_KEYS)})")
    print("=" * 70)

    config = load_config()
    cfg    = config['qdecomp_ppo']

    total_timesteps = total_timesteps_override or cfg.get('total_timesteps', 3_000_000)
    eval_freq       = cfg.get('eval_freq',       25_000)
    n_eval_episodes = cfg.get('n_eval_episodes', 5)
    save_freq       = cfg.get('save_freq',       100_000)
    n_steps         = cfg.get('n_steps',         2048)

    timestamp  = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    output_dir = project_root / "logs" / "qdecomp_ppo" / f"go2_qdecomp_ppo_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yml", 'w') as f:
        yaml.dump(config, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Output: {output_dir}\n")

    env      = make_env(config)
    eval_env = make_env(config)

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent  = DecompPPO(obs_dim, action_dim, config, device)
    writer = SummaryWriter(str(output_dir / "tensorboard"))

    total_steps      = 0
    update_num       = 0
    best_eval_reward = -np.inf

    print(f"Total timesteps: {total_timesteps:,}  |  n_steps per update: {n_steps:,}\n")

    while total_steps < total_timesteps:
        rollout_metrics = agent.collect_rollout(env)
        total_steps    += n_steps
        update_num     += 1

        update_metrics = agent.update()

        for k, v in {**rollout_metrics, **update_metrics}.items():
            writer.add_scalar(k, v, total_steps)

        if update_num % 5 == 0:
            print(f"  Step {total_steps:>8,} | update={update_num:>4} | "
                  f"ep_r={rollout_metrics['rollout/mean_ep_reward']:.2f} | "
                  f"loss_a={update_metrics['loss/actor']:.4f}")

        if total_steps % eval_freq < n_steps:
            metrics = evaluate(agent, eval_env, n_eval_episodes)
            for k, v in metrics.items():
                writer.add_scalar(k, v, total_steps)
            mean_r = metrics['eval/mean_reward']
            print(f"\n[Eval @ {total_steps:,}] mean_r={mean_r:.2f}  "
                  f"success={metrics['eval/success_rate']:.0%}\n")
            if mean_r > best_eval_reward:
                best_eval_reward = mean_r
                torch.save(
                    {'policy': agent.policy.state_dict(), 'step': total_steps},
                    output_dir / "best_model.pt",
                )

        if total_steps % save_freq < n_steps:
            torch.save(
                {'policy': agent.policy.state_dict(), 'step': total_steps},
                output_dir / f"checkpoint_{total_steps}.pt",
            )

    torch.save(
        {'policy': agent.policy.state_dict(), 'step': total_steps},
        output_dir / "final_model.pt",
    )
    print(f"\nTraining complete. Saved to {output_dir}")
    writer.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=None)
    args = parser.parse_args()
    train(total_timesteps_override=args.timesteps)
