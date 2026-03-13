"""
Train SAC agent for Go2 self-recovery
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
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import torch

from environment.go2_env import Go2Env
from src.utils.callbacks import RewardLoggerCallback, CurriculumMonitorCallback, TensorBoardMetricsCallback


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


ACTIVATION_FNS = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "elu": torch.nn.ELU,
}


def train():
    """Main training function"""
    
    print("=" * 70)
    print("SAC Training - Go2 Self-Recovery")
    print("=" * 70)
    
    # Load config
    print("\n1. Loading configuration...")
    config = load_config()
    sac_config = config['sac']
    print("   ✓ Config loaded")
    
    # Create output directory
    timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    run_name = f"go2_{timestamp}"
    output_dir = project_root / "logs" / "sac" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"   ✓ Output directory: {output_dir}")
    
    # Save config
    with open(output_dir / "config.yml", 'w') as f:
        yaml.dump(config, f)
    
    # Create environments
    # SAC is off-policy so fewer parallel envs are typical; use 1 for standard training
    print("\n2. Creating environment...")
    n_envs = config.get('training', {}).get('num_parallel_envs', 1)
    # SAC works best with a single env or a small number
    n_envs = min(n_envs, 4)
    print(f"   Creating {n_envs} parallel environment(s)...")
    
    if n_envs > 1:
        env = SubprocVecEnv([make_env(config, i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(config, 0)])
    print("   ✓ Environment created")
    
    # Create eval environment
    print("\n3. Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(config, 0)])
    print("   ✓ Eval environment created")
    
    # Setup callbacks
    print("\n4. Setting up callbacks...")
    
    # Reward logger
    reward_logger = RewardLoggerCallback(log_freq=5)
    
    # Curriculum monitor
    curriculum_monitor = CurriculumMonitorCallback(log_freq=100)
    
    # TensorBoard metrics
    tensorboard_callback = TensorBoardMetricsCallback(log_freq=1000)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config['training']['save_freq'] // n_envs, 1),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="sac_go2"
    )
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=max(config['training']['eval_freq'] // n_envs, 1),
        n_eval_episodes=config['training']['n_eval_episodes'],
        deterministic=True,
        render=False
    )
    
    callback_list = CallbackList([
        reward_logger, 
        curriculum_monitor,
        tensorboard_callback,
        checkpoint_callback, 
        eval_callback
    ])
    print("   ✓ Callbacks configured")
    
    # Resolve activation function
    act_fn_name = sac_config['policy_kwargs'].get('activation_fn', 'relu')
    activation_fn = ACTIVATION_FNS.get(act_fn_name, torch.nn.ReLU)
    
    # Resolve entropy coefficient (can be "auto" or a float)
    ent_coef = sac_config['ent_coef']
    if isinstance(ent_coef, str) and ent_coef == "auto":
        ent_coef = "auto"
    else:
        ent_coef = float(ent_coef)
    
    # Resolve target entropy
    target_entropy = sac_config.get('target_entropy', 'auto')
    if isinstance(target_entropy, str) and target_entropy == "auto":
        target_entropy = "auto"
    else:
        target_entropy = float(target_entropy)
    
    # Create SAC model
    print("\n5. Creating SAC model...")
    print(f"   Architecture: {sac_config['policy_kwargs']['net_arch']}")
    print(f"   Learning rate: {sac_config['learning_rate']}")
    print(f"   Buffer size: {sac_config['buffer_size']:,}")
    print(f"   Batch size: {sac_config['batch_size']}")
    print(f"   Tau: {sac_config['tau']}")
    print(f"   Entropy coef: {ent_coef}")
    print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=sac_config['learning_rate'],
        buffer_size=sac_config['buffer_size'],
        learning_starts=sac_config['learning_starts'],
        batch_size=sac_config['batch_size'],
        tau=sac_config['tau'],
        gamma=sac_config['gamma'],
        train_freq=sac_config['train_freq'],
        gradient_steps=sac_config['gradient_steps'],
        ent_coef=ent_coef,
        target_update_interval=sac_config['target_update_interval'],
        target_entropy=target_entropy,
        policy_kwargs={
            'net_arch': sac_config['policy_kwargs']['net_arch'],
            'activation_fn': activation_fn,
        },
        tensorboard_log=str(output_dir / "tensorboard"),
        verbose=1,
    )
    print("   ✓ Model created")
    
    # Training parameters
    total_timesteps = config['training']['total_timesteps']
    print(f"\n6. Starting training...")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Learning starts after: {sac_config['learning_starts']:,} steps")
    print("\n" + "=" * 70)
    print("Training in progress... (Check TensorBoard for live metrics)")
    print(f"   tensorboard --logdir {output_dir / 'tensorboard'}")
    print("=" * 70 + "\n")
    
    try:
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True,
            log_interval=config['training'].get('log_interval', 4),
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
