"""
Visualize Go2 Self-Recovery Environment with MuJoCo Viewer
"""

import sys
from pathlib import Path
import numpy as np
import yaml
import time
import mujoco
import mujoco.viewer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from environment.go2_env import Go2Env


def load_config():
    """Load config from YAML"""
    config_path = Path(__file__).parent / "config" / "train_config.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def visualize():
    """Visualize environment with interactive viewer"""
    
    print("=" * 60)
    print("Go2 Self-Recovery Environment - Interactive Visualization")
    print("=" * 60)
    print("\nLoading environment...")
    
    # Load config
    config = load_config()
    
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
    
    # Create environment WITHOUT rendering (we'll handle it manually)
    env = Go2Env(config, render_mode=None)
    
    print("✓ Environment loaded")
    print("\nVisualization Controls:")
    print("  - Double-click: Rotate camera")
    print("  - Right-click + drag: Pan camera")
    print("  - Scroll: Zoom in/out")
    print("  - Close window or press Ctrl+C to stop\n")
    
    # Launch viewer manually
    print("Opening MuJoCo viewer...")
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print("✓ Viewer opened\n")
        
        try:
            episode = 1
            while viewer.is_running():
                print(f"\n{'='*40}")
                print(f"Episode {episode}")
                print(f"{'='*40}")
                
                # Reset environment
                obs, info = env.reset()
                print(f"Initial height: {info['base_height']:.3f}m")
                
                episode_reward = 0
                step = 0
                
                # Run episode
                while step < 500 and viewer.is_running():
                    # Random action (replace with trained policy later)
                    action = env.action_space.sample()
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    step += 1
                    
                    # Sync viewer
                    viewer.sync()
                    
                    # Print progress every 50 steps
                    if step % 50 == 0:
                        height = info['base_height']
                        orientation = obs[26]  # R_g from observation
                        print(f"  Step {step:3d} | Height: {height:.3f}m | "
                              f"Orient: {orientation:.3f} | Reward: {reward:.3f}")
                    
                    # Small delay for smoother visualization
                    time.sleep(0.02)
                    
                    if terminated or truncated:
                        break
                
                if not viewer.is_running():
                    break
                    
                print(f"\nEpisode finished:")
                print(f"  - Steps: {step}")
                print(f"  - Total reward: {episode_reward:.2f}")
                print(f"  - Final height: {info['base_height']:.3f}m")
                
                episode += 1
                
                # Wait a bit before next episode
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\n\nVisualization stopped by user")
        finally:
            env.close()
            print("Environment closed")


if __name__ == "__main__":
    visualize()
