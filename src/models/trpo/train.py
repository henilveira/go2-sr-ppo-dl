"""
Train TRPO agent for Go2 self-recovery.
Uses TRPO with GAE for advantage estimation.
"""

import sys
from pathlib import Path
import yaml
from datetime import datetime
import importlib

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import torch

from environment.go2_env import Go2Env
from src.utils.callbacks import (
    RewardLoggerCallback,
    CurriculumMonitorCallback,
    TensorBoardMetricsCallback,
    TerrainCurriculumCallback,
)


def _load_trpo_class():
    """Load TRPO class from sb3-contrib with a clear failure message."""
    try:
        return importlib.import_module("sb3_contrib").TRPO
    except Exception as exc:
        raise ImportError(
            "TRPO training requires sb3-contrib. Install with: pip install sb3-contrib"
        ) from exc


def load_config():
    """Load configuration from YAML."""
    config_path = project_root / "config" / "train_config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_trpo_config(config):
    """Resolve TRPO config profile while keeping backward compatibility."""
    if "trpo" in config:
        return "trpo", config["trpo"]

    profile_name = config.get("training", {}).get("trpo_profile", "trpo")
    if profile_name not in config:
        available_profiles = [key for key in config.keys() if key.endswith("trpo")]
        raise KeyError(
            f"TRPO profile '{profile_name}' not found in config. "
            f"Available profiles: {available_profiles}"
        )

    return profile_name, config[profile_name]


def make_env(config, rank=0):
    """Factory function to create a single environment."""

    def _init():
        # Keep defaults for robustness when loading partial configs.
        if "training" not in config:
            config["training"] = {
                "drop_height": 0.3,
                "random_orientation": True,
                "random_joint_positions": True,
                "max_episode_steps": 1024,
            }

        if "simulation" not in config:
            config["simulation"] = {
                "n_substeps": 5,
            }

        env = Go2Env(config, render_mode=None)
        env = Monitor(env)
        return env

    return _init


def train():
    """Main TRPO training function."""
    TRPO = _load_trpo_class()

    print("=" * 70)
    print("TRPO Training - Go2 Self-Recovery")
    print("=" * 70)

    print("\n1. Loading configuration...")
    config = load_config()
    trpo_profile, trpo_config = get_trpo_config(config)
    print("   ✓ Config loaded")
    print(f"   ✓ TRPO profile: {trpo_profile}")

    timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    run_name = f"TRPO_{timestamp}"
    output_dir = project_root / "logs" / "trpo" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"   ✓ Output directory: {output_dir}")

    with open(output_dir / "config.yml", "w") as f:
        yaml.dump(config, f)

    print("\n2. Creating environments...")
    n_envs = config.get("training", {}).get("num_parallel_envs", 12)
    print(f"   Creating {n_envs} parallel environments...")
    env = SubprocVecEnv([make_env(config, i) for i in range(n_envs)])
    print("   ✓ Environments created")

    print("\n3. Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(config, 0)])
    print("   ✓ Eval environment created")

    print("\n4. Setting up callbacks...")
    training_config = config.get("training", {})
    total_timesteps = training_config.get(
        "total_timesteps_trpo",
        training_config.get("total_timesteps"),
    )
    if total_timesteps is None:
        raise KeyError("Missing training.total_timesteps_trpo (or legacy training.total_timesteps)")

    reward_logger = RewardLoggerCallback(log_freq=5)
    curriculum_monitor = CurriculumMonitorCallback(log_freq=100)
    tensorboard_callback = TensorBoardMetricsCallback(log_freq=1000)

    terrain_curriculum_callback = TerrainCurriculumCallback(
        total_timesteps=total_timesteps,
        log_freq=10_000,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=config["training"]["save_freq"] // n_envs,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="trpo_go2",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=config["training"]["eval_freq"] // n_envs,
        n_eval_episodes=config["training"]["n_eval_episodes"],
        deterministic=True,
        render=False,
    )

    callback_list = CallbackList([
        reward_logger,
        curriculum_monitor,
        terrain_curriculum_callback,
        tensorboard_callback,
        checkpoint_callback,
        eval_callback,
    ])
    print("   ✓ Callbacks configured")

    print("\n5. Creating TRPO model...")
    print(f"   Architecture: {trpo_config['policy_kwargs']['net_arch']}")
    print(f"   Learning rate: {trpo_config['learning_rate']}")
    print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    model = TRPO(
        "MlpPolicy",
        env,
        learning_rate=trpo_config["learning_rate"],
        n_steps=trpo_config["n_steps"],
        batch_size=trpo_config["batch_size"],
        gamma=trpo_config["gamma"],
        gae_lambda=trpo_config["gae_lambda"],
        cg_max_steps=trpo_config["cg_max_steps"],
        cg_damping=trpo_config["cg_damping"],
        line_search_shrinking_factor=trpo_config["line_search_shrinking_factor"],
        line_search_max_iter=trpo_config["line_search_max_iter"],
        n_critic_updates=trpo_config["n_critic_updates"],
        target_kl=trpo_config["target_kl"],
        policy_kwargs={
            "net_arch": trpo_config["policy_kwargs"]["net_arch"],
            "activation_fn": torch.nn.ReLU,
        },
        tensorboard_log=str(output_dir / "tensorboard"),
        verbose=int(trpo_config.get("verbose", 1)),
        device="auto",
    )
    print("   ✓ Model created")

    print("\n6. Starting training...")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Expected updates: ~{total_timesteps // trpo_config['n_steps']:,}")
    print(f"   Estimated time: ~{total_timesteps / (n_envs * 450):.0f} minutes")
    print("\n" + "=" * 70)
    print("Training in progress... (Check TensorBoard for live metrics)")
    print(f"   tensorboard --logdir {output_dir / 'tensorboard'}")
    print("=" * 70 + "\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True,
        )

        final_model_path = output_dir / "final_model"
        model.save(final_model_path)
        print("\n✓ Training complete!")
        print(f"   Final model saved to: {final_model_path}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        interrupted_model_path = output_dir / "interrupted_model"
        model.save(interrupted_model_path)
        print(f"Model saved to: {interrupted_model_path}")

    finally:
        env.close()
        eval_env.close()

    print("\n" + "=" * 70)
    print("Training session complete!")
    print("=" * 70)
    print(f"\nResults saved in: {output_dir}")


if __name__ == "__main__":
    train()
