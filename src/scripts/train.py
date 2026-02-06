"""
Train PPO agent for Go2 self-recovery
Based on paper: "Self-Recovery of Quadrupedal Robot Using DRL" (2024)
"""

import sys
from pathlib import Path
import yaml
import numpy as np
from datetime import datetime
import os

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import Stable-Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import torch

from environment.go2_env import Go2Env
from src.utils.callbacks import RewardLoggerCallback, CurriculumMonitorCallback


def load_config():
    """Load configuration from YAML"""
    config_path = project_root / "config" / "train_config.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def make_env(config, rank=0):
    """Factory function to create a single environment"""
    def _init():
        # Add training config
        if 'training' not in config:
            config['training'] = {
                'drop_height': 0.3,
                'random_orientation': True,
                'random_joint_positions': True,
                'max_episode_steps': 1024
            }
        
        if 'simulation' not in config:
            config['simulation'] = {
                'n_substeps': 5
            }
        
        env = Go2Env(config, render_mode=None)
        env = Monitor(env)  # Wrap for logging
        return env
    return _init


def train():
    """Main training function"""
    
    print("=" * 70)
    print("PPO Training - Go2 Self-Recovery")
    print("=" * 70)
    
    # Load config
    print("\n1. Loading configuration...")
    config = load_config()
    ppo_config = config['ppo']
    print("   ✓ Config loaded")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"go2_self_recovery_{timestamp}"
    output_dir = project_root / "logs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"   ✓ Output directory: {output_dir}")
    
    # Save config
    with open(output_dir / "config.yml", 'w') as f:
        yaml.dump(config, f)
    
    # Create vectorized environments (parallel training)
    print("\n2. Creating environments...")
    n_envs = config.get('training', {}).get('num_parallel_envs', 12)
    print(f"   Creating {n_envs} parallel environments...")
    
    # Use SubprocVecEnv for true parallelism (faster)
    env = SubprocVecEnv([make_env(config, i) for i in range(n_envs)])
    print("   ✓ Environments created")
    
    # Create eval environment
    print("\n3. Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(config, 0)])
    print("   ✓ Eval environment created")
    
    # Setup callbacks
    print("\n4. Setting up callbacks...")
    
    # Reward logger - show stats in terminal
    reward_logger = RewardLoggerCallback(log_freq=5)  # Log every 5 rollouts
    
    # Curriculum monitor
    curriculum_monitor = CurriculumMonitorCallback(log_freq=100)
    
    # Checkpoint callback - save model every N steps
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_freq'] // n_envs,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_go2"
    )
    
    # Evaluation callback - evaluate periodically
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=config['training']['eval_freq'] // n_envs,
        n_eval_episodes=config['training']['n_eval_episodes'],
        deterministic=True,
        render=False
    )
    
    callback_list = CallbackList([
        reward_logger, 
        curriculum_monitor, 
        checkpoint_callback, 
        eval_callback
    ])
    print("   ✓ Callbacks configured")
    
    # Create PPO model
    print("\n5. Creating PPO model...")
    print(f"   Architecture: {ppo_config['policy_kwargs']['net_arch']}")
    print(f"   Learning rate: {ppo_config['learning_rate']}")
    print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=ppo_config['learning_rate'],
        n_steps=ppo_config['n_steps'],
        batch_size=ppo_config['batch_size'],
        n_epochs=ppo_config['n_epochs'],
        gamma=ppo_config['gamma'],
        gae_lambda=ppo_config['gae_lambda'],
        clip_range=ppo_config['clip_range'],
        ent_coef=ppo_config['ent_coef'],
        vf_coef=ppo_config['vf_coef'],
        max_grad_norm=ppo_config['max_grad_norm'],
        policy_kwargs={
            'net_arch': ppo_config['policy_kwargs']['net_arch'],
            'activation_fn': torch.nn.ReLU
        },
        tensorboard_log=str(output_dir / "tensorboard"),
        verbose=1
    )
    print("   ✓ Model created")
    
    # Training parameters
    total_timesteps = config['training']['total_timesteps']
    print(f"\n6. Starting training...")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Expected episodes: ~{total_timesteps // ppo_config['n_steps']:,}")
    print(f"   Estimated time: ~{total_timesteps / (n_envs * 500):.0f} minutes")
    print("\n" + "=" * 70)
    print("Training in progress... (Check TensorBoard for live metrics)")
    print(f"   tensorboard --logdir {output_dir / 'tensorboard'}")
    print("=" * 70 + "\n")
    
    try:
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True
        )
        
        # Save final model
        final_model_path = output_dir / "final_model"
        model.save(final_model_path)
        print(f"\n✓ Training complete!")
        print(f"   Final model saved to: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        interrupted_model_path = output_dir / "interrupted_model"
        model.save(interrupted_model_path)
        print(f"Model saved to: {interrupted_model_path}")
    
    finally:
        # Cleanup
        env.close()
        eval_env.close()
    
    print("\n" + "=" * 70)
    print("Training session complete!")
    print("=" * 70)
    print(f"\nResults saved in: {output_dir}")
    print("\nNext steps:")
    print("  1. Check TensorBoard for training curves")
    print("  2. Evaluate best model with: python scripts/evaluate.py")
    print("  3. Visualize trained policy")


if __name__ == "__main__":
    train()
