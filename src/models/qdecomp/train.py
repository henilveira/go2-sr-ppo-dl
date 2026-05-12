"""
Q-Decomposition SAC training for Go2 self-recovery.

Usage:
    python src/models/qdecomp/train.py
    python src/models/qdecomp/train.py --timesteps 50000   # quick test
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
from src.models.qdecomp.agent import QDecompSAC
from src.models.qdecomp.networks import REWARD_KEYS


def load_config():
    config_path = project_root / "config" / "train_config.yml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_env(config):
    return Go2Env(config, render_mode=None)


def evaluate(agent: QDecompSAC, env: Go2Env, n_episodes: int = 5) -> dict:
    total_rewards = []
    success_count = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_rewards.append(ep_reward)
        breakdown = info.get('reward_breakdown', {})
        if breakdown.get('R_g', 0.0) > 0.8:
            success_count += 1

    return {
        'eval/mean_reward':  float(np.mean(total_rewards)),
        'eval/std_reward':   float(np.std(total_rewards)),
        'eval/success_rate': success_count / n_episodes,
    }


def train(total_timesteps_override: int | None = None):
    print("=" * 70)
    print("Q-Decomposition SAC — Go2 Self-Recovery")
    print(f"  Subagents: {len(REWARD_KEYS)} ({', '.join(REWARD_KEYS)})")
    print("=" * 70)

    config = load_config()
    cfg = config['qdecomp']

    total_timesteps = total_timesteps_override or cfg.get('total_timesteps', 3_000_000)
    eval_freq       = cfg.get('eval_freq', 25_000)
    n_eval_episodes = cfg.get('n_eval_episodes', 5)
    save_freq       = cfg.get('save_freq', 100_000)

    timestamp  = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    run_name   = f"go2_qdecomp_{timestamp}"
    output_dir = project_root / "logs" / "qdecomp" / run_name
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

    agent = QDecompSAC(obs_dim, action_dim, config, device)

    writer = SummaryWriter(str(output_dir / "tensorboard"))

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_steps  = 0
    episode_num    = 0
    best_eval_reward = -np.inf

    print(f"Starting training for {total_timesteps:,} timesteps...")
    print(f"  Learning starts after: {cfg.get('learning_starts', 5000):,} steps\n")

    for t in range(1, total_timesteps + 1):
        if t < cfg.get('learning_starts', 5_000):
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        reward_breakdown = info.get('reward_breakdown', {})
        agent.store_transition(obs, action, reward_breakdown, next_obs, done)

        obs             = next_obs
        episode_reward += reward
        episode_steps  += 1

        if done:
            writer.add_scalar('train/episode_reward', episode_reward, t)
            writer.add_scalar('train/episode_length', episode_steps, t)
            for key in REWARD_KEYS:
                writer.add_scalar(f'train/reward_{key}', reward_breakdown.get(key, 0.0), t)

            episode_num    += 1
            episode_reward  = 0.0
            episode_steps   = 0
            obs, _          = env.reset()

            if episode_num % 10 == 0:
                print(f"  Step {t:>8,} | Episode {episode_num:>5} | buf={len(agent.buffer):,}")

        if t >= cfg.get('learning_starts', 5_000) and t % cfg.get('train_freq', 1) == 0:
            metrics = agent.update()
            for k, v in metrics.items():
                writer.add_scalar(k, v, t)

        if t % eval_freq == 0:
            eval_metrics = evaluate(agent, eval_env, n_eval_episodes)
            for k, v in eval_metrics.items():
                writer.add_scalar(k, v, t)
            mean_r = eval_metrics['eval/mean_reward']
            print(f"\n[Eval @ {t:,}] mean_reward={mean_r:.2f} "
                  f"success={eval_metrics['eval/success_rate']:.0%}\n")
            if mean_r > best_eval_reward:
                best_eval_reward = mean_r
                torch.save({
                    'actor':   agent.actor.state_dict(),
                    'critics': agent.critics.state_dict(),
                    'step':    t,
                }, output_dir / "best_model.pt")

        if t % save_freq == 0:
            torch.save({
                'actor':   agent.actor.state_dict(),
                'critics': agent.critics.state_dict(),
                'step':    t,
            }, output_dir / f"checkpoint_{t}.pt")

    torch.save({
        'actor':   agent.actor.state_dict(),
        'critics': agent.critics.state_dict(),
        'step':    total_timesteps,
    }, output_dir / "final_model.pt")
    print(f"\nTraining complete. Models saved to {output_dir}")

    writer.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=None,
                        help="Override total_timesteps from config")
    args = parser.parse_args()
    train(total_timesteps_override=args.timesteps)
