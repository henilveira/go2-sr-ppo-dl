"""
Test script to visualize rewards and understand robot behavior
Run this to see what the robot is doing and if rewards make sense
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import time

def load_config():
    config_path = project_root / "config" / "train_config.yml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_random_actions():
    """Test environment with random actions to see reward distribution"""
    from environment.go2_env import Go2Env
    
    print("=" * 70)
    print("Testing Environment with Random Actions")
    print("=" * 70)
    
    config = load_config()
    config['training']['max_episode_steps'] = 200  # Short episodes for testing
    
    env = Go2Env(config, render_mode=None)
    
    total_rewards = []
    reward_components = {
        'R_h': [], 'R_g': [], 'R_ad': [], 'R_v': [], 'R_vb': [],
        'R_h_cl': [], 'R_jp': [], 'R_fc': [],
        'R_alive': [], 'R_progress': [], 'curriculum_active': []
    }
    heights = []
    orientations = []
    
    n_episodes = 10
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(200):
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Collect stats
            if 'reward_breakdown' in info:
                rb = info['reward_breakdown']
                for key in reward_components:
                    if key in rb:
                        reward_components[key].append(rb[key])
            
            heights.append(info['base_height'])
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {ep+1}: Total Reward = {episode_reward:.2f}")
    
    env.close()
    
    print("\n" + "=" * 70)
    print("REWARD STATISTICS (Random Policy)")
    print("=" * 70)
    print(f"\nTotal Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Height: {np.mean(heights):.3f} ± {np.std(heights):.3f} (target: 0.31)")
    
    print("\nReward Components (mean ± std):")
    for key, values in reward_components.items():
        if len(values) > 0:
            print(f"  {key:15s}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    print(f"\nCurriculum Active: {np.mean(reward_components['curriculum_active'])*100:.1f}% of steps")
    
    return total_rewards


def test_standing_pose():
    """Test if standing pose gives high reward"""
    from environment.go2_env import Go2Env
    import mujoco
    
    print("\n" + "=" * 70)
    print("Testing Standing Pose Reward")
    print("=" * 70)
    
    config = load_config()
    env = Go2Env(config, render_mode=None)
    
    # Reset and manually set to standing pose
    obs, info = env.reset()
    
    # Set robot to upright position
    env.data.qpos[0:3] = [0, 0, 0.34]  # Position at target height
    env.data.qpos[3:7] = [1, 0, 0, 0]  # Upright quaternion (w, x, y, z)
    
    # Set joints to standing pose
    standing_joints = np.array([
        0.0, 0.67, -1.3,   # FR
        0.0, 0.67, -1.3,   # FL
        0.0, 0.67, -1.3,   # RR
        0.0, 0.67, -1.3    # RL
    ])
    env.data.qpos[7:19] = standing_joints
    env.data.qvel[:] = 0  # Zero velocity
    
    mujoco.mj_forward(env.model, env.data)
    
    # Get observation and compute reward
    obs = env._get_observation()
    action = np.zeros(12)  # No action
    reward, reward_info = env._compute_reward(obs, action)
    
    print(f"\nWhen standing upright:")
    print(f"  Total Reward: {reward:.4f}")
    print(f"  Height: {env.data.qpos[2]:.3f}")
    print("\nReward breakdown:")
    for key, value in reward_info.items():
        if key != 'total':
            print(f"  {key:15s}: {value:.4f}")
    
    env.close()
    
    return reward


def test_fallen_pose():
    """Test reward when robot is fallen"""
    from environment.go2_env import Go2Env
    import mujoco
    
    print("\n" + "=" * 70)
    print("Testing Fallen Pose Reward")
    print("=" * 70)
    
    config = load_config()
    env = Go2Env(config, render_mode=None)
    
    # Reset - robot starts fallen
    obs, info = env.reset()
    
    # Ensure robot is on its back
    env.data.qpos[0:3] = [0, 0, 0.1]  # Low position
    # Upside down quaternion (180 degrees roll)
    quat = env._euler_to_quat([np.pi, 0, 0])
    env.data.qpos[3:7] = quat
    env.data.qvel[:] = 0
    
    mujoco.mj_forward(env.model, env.data)
    
    # Get observation and compute reward
    obs = env._get_observation()
    action = np.zeros(12)
    reward, reward_info = env._compute_reward(obs, action)
    
    print(f"\nWhen fallen (on back):")
    print(f"  Total Reward: {reward:.4f}")
    print(f"  Height: {env.data.qpos[2]:.3f}")
    print("\nReward breakdown:")
    for key, value in reward_info.items():
        if key != 'total':
            print(f"  {key:15s}: {value:.4f}")
    
    env.close()
    
    return reward


if __name__ == "__main__":
    standing_reward = test_standing_pose()
    fallen_reward = test_fallen_pose()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Standing reward: {standing_reward:.4f}")
    print(f"Fallen reward:   {fallen_reward:.4f}")
    print(f"Reward gap:      {standing_reward - fallen_reward:.4f}")
    print()
    if standing_reward > fallen_reward * 2:
        print("✓ Good! Standing gives significantly more reward than fallen.")
    else:
        print("⚠ Warning: Reward gap might be too small for effective learning.")
    
    print("\n" + "=" * 70)
    test_random_actions()
