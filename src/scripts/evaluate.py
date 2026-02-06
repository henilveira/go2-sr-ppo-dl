"""
Evaluate trained PPO model with visualization
"""

import sys
from pathlib import Path
import yaml
import time
import mujoco.viewer
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from environment.go2_env import Go2Env


def load_config():
    """Load configuration from YAML"""
    config_path = project_root / "config" / "train_config.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Ensure required sections exist (do not override user settings)
    config.setdefault('training', {
        'drop_height': 0.5,
        'random_orientation': True,
        'random_joint_positions': True,
        'max_episode_steps': 1024
    })
    config.setdefault('simulation', {'n_substeps': 10})

    return config


def find_latest_model():
    """Find the most recent trained model"""
    logs_dir = project_root / "logs"
    
    if not logs_dir.exists():
        return None
    
    # Look for directories with models
    model_dirs = []
    for run_dir in logs_dir.iterdir():
        if run_dir.is_dir():
            # Check for best model
            best_model = run_dir / "best_model" / "best_model.zip"
            if best_model.exists():
                model_dirs.append((run_dir.stat().st_mtime, best_model, "best"))
            
            # Check for final model
            final_model = run_dir / "final_model.zip"
            if final_model.exists():
                model_dirs.append((run_dir.stat().st_mtime, final_model, "final"))
    
    if not model_dirs:
        return None
    
    # Return most recent
    model_dirs.sort(reverse=True)
    return model_dirs[0][1], model_dirs[0][2]


def evaluate():
    """Evaluate trained model with visualization"""
    
    print("=" * 70)
    print("PPO Model Evaluation - Go2 Self-Recovery")
    print("=" * 70)
    
    # Load config
    print("\n1. Loading configuration...")
    config = load_config()
    print("   ✓ Config loaded")
    
    # Find model
    print("\n2. Loading trained model...")
    model_info = find_latest_model()
    
    if model_info is None:
        print("   ✗ No trained model found!")
        print("\n   Please train a model first:")
        print("   python scripts/train.py")
        return
    
    model_path, model_type = model_info
    print(f"   ✓ Found {model_type} model: {model_path.parent.parent.name}")
    
    try:
        model = PPO.load(model_path)
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return
    
    # Create environment
    print("\n3. Creating environment...")
    env = Go2Env(config, render_mode=None)
    print("   ✓ Environment created")
    
    print("\n4. Starting evaluation...")
    print("\nControls:")
    print("  - Double-click + drag: Rotate camera")
    print("  - Right-click + drag: Pan camera")
    print("  - Scroll: Zoom in/out")
    print("  - Close window or Ctrl+C to stop\n")
    
    # Launch viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        try:
            episode = 1
            total_episodes = 5  # Run 5 episodes
            
            while episode <= total_episodes and viewer.is_running():
                print(f"\n{'='*50}")
                print(f"Episode {episode}/{total_episodes}")
                print(f"{'='*50}")
                
                # Reset environment
                obs, info = env.reset()
                print(f"Initial height: {info['base_height']:.3f}m")
                
                episode_reward = 0
                step = 0
                done = False
                
                # Run episode with trained policy
                while not done and viewer.is_running():
                    # Get action from trained policy
                    action, _states = model.predict(obs, deterministic=True)
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    step += 1
                    
                    # Sync viewer
                    viewer.sync()
                    
                    # Print progress every 100 steps
                    if step % 100 == 0:
                        height = info['base_height']
                        print(f"  Step {step:3d} | Height: {height:.3f}m | Reward: {reward:.3f}")
                    
                    # Small delay for smooth visualization
                    time.sleep(0.02)
                    
                    done = terminated or truncated
                
                if not viewer.is_running():
                    break
                
                print(f"\nEpisode Results:")
                print(f"  - Steps: {step}")
                print(f"  - Total reward: {episode_reward:.2f}")
                print(f"  - Final height: {info['base_height']:.3f}m")
                
                # Check if successful recovery
                if info['base_height'] > 0.25:
                    print(f"  ✓ SUCCESS! Robot recovered to standing position")
                else:
                    print(f"  ✗ Failed to fully recover")
                
                episode += 1
                
                if episode <= total_episodes:
                    print("\nStarting next episode in 2 seconds...")
                    time.sleep(2.0)
                
        except KeyboardInterrupt:
            print("\n\nEvaluation stopped by user")
        finally:
            env.close()
    
    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()
