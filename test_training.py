"""
Quick test to verify training setup works
Runs a short training session (1000 steps)
"""

import sys
from pathlib import Path
import yaml
import torch

# Make sure we import from quadrupedal_sr, not root
sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from environment.go2_env import Go2Env


def load_config():
    config_path = Path(__file__).parent / "config" / "train_config.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add training config
    config['training'] = {
        'drop_height': 0.3,
        'random_orientation': True,
        'random_joint_positions': True,
        'max_episode_steps': 1024
    }
    config['simulation'] = {'n_substeps': 5}
    
    return config


print("=" * 60)
print("Quick Training Test (1000 steps)")
print("=" * 60)

print("\n1. Loading config...")
config = load_config()
print("   ✓ Config loaded")

print("\n2. Creating environment...")
env = Go2Env(config, render_mode=None)
env = Monitor(env)
env = DummyVecEnv([lambda: env])
print("   ✓ Environment created")

print("\n3. Creating PPO model...")
print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=config['ppo']['learning_rate'],
    n_steps=256,  # Smaller for quick test
    batch_size=32,
    n_epochs=4,
    policy_kwargs={
        'net_arch': config['ppo']['policy_kwargs']['net_arch'],
        'activation_fn': torch.nn.ReLU
    },
    verbose=1
)
print("   ✓ Model created")

print("\n4. Running training test (1000 steps)...")
print("-" * 60)

try:
    model.learn(total_timesteps=1000, progress_bar=True)
    print("-" * 60)
    print("\n✓ Training test PASSED!")
    print("\nEverything is working correctly.")
    print("Ready to start full training with: python scripts/train.py")
    
except Exception as e:
    print(f"\n✗ Training test FAILED: {e}")
    import traceback
    traceback.print_exc()

finally:
    env.close()
