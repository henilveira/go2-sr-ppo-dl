"""
Debug script to understand reward behavior
Run this to see what rewards the robot gets in different states
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from environment.go2_env import Go2Env
import yaml


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "train_config.yml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_rewards_in_different_states():
    """Test what rewards the robot gets in various states"""
    
    config = load_config()
    env = Go2Env(config)
    
    print("=" * 70)
    print("REWARD DEBUG - Testing rewards in different robot states")
    print("=" * 70)
    
    # Run several episodes and collect data
    n_episodes = 5
    steps_per_episode = 50
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        
        print(f"\n{'='*70}")
        print(f"EPISODE {ep + 1}")
        print(f"{'='*70}")
        
        # Initial state analysis
        initial_height = info['base_height']
        theta_B = obs[24:27]  # Before normalization we need raw
        print(f"\nInitial state:")
        print(f"  Height: {initial_height:.3f}m")
        print(f"  θ_B (body gravity): {theta_B}")
        print(f"  Feet contacts: {info['feet_contacts']}")
        
        total_rewards = []
        r_h_list, r_g_list, r_ad_list = [], [], []
        
        for step in range(steps_per_episode):
            # Random action
            action = env.action_space.sample()
            
            # Gradually reduce randomness to test stable behavior
            action = action * 0.3  # Smaller actions
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if 'reward_breakdown' in info:
                rb = info['reward_breakdown']
                total_rewards.append(reward)
                r_h_list.append(rb['R_h'])
                r_g_list.append(rb['R_g'])
                r_ad_list.append(rb['R_ad'])
                
                if step % 10 == 0:
                    print(f"\n  Step {step}:")
                    print(f"    Height: {info['base_height']:.3f}m | R_h: {rb['R_h']:.3f}")
                    print(f"    θ_B: [{obs[24]:.2f}, {obs[25]:.2f}, {obs[26]:.2f}] | R_g: {rb['R_g']:.3f}")
                    print(f"    R_ad: {rb['R_ad']:.3f} | Curriculum: {rb['curriculum_active']}")
                    print(f"    Total: {reward:.3f}")
            
            if terminated or truncated:
                break
        
        # Summary
        print(f"\n  Episode summary:")
        print(f"    Avg total reward: {np.mean(total_rewards):.3f}")
        print(f"    Avg R_h: {np.mean(r_h_list):.3f}")
        print(f"    Avg R_g: {np.mean(r_g_list):.3f}")
        print(f"    Avg R_ad: {np.mean(r_ad_list):.3f}")
    
    env.close()


def test_action_effects():
    """Test how different actions affect the robot"""
    
    config = load_config()
    env = Go2Env(config)
    
    print("\n" + "=" * 70)
    print("ACTION EFFECTS TEST")
    print("=" * 70)
    
    obs, info = env.reset()
    
    print("\nInitial state:")
    print(f"  Height: {info['base_height']:.3f}m")
    
    # Test 1: Zero action (stay still)
    print("\n1. Zero action (10 steps):")
    for i in range(10):
        action = np.zeros(12)
        obs, reward, _, _, info = env.step(action)
    print(f"   Final height: {info['base_height']:.3f}m")
    print(f"   Final reward: {reward:.3f}")
    
    # Reset
    obs, info = env.reset()
    
    # Test 2: Extreme actions
    print("\n2. Extreme action +1 (10 steps):")
    for i in range(10):
        action = np.ones(12)
        obs, reward, _, _, info = env.step(action)
    print(f"   Final height: {info['base_height']:.3f}m")
    print(f"   Final reward: {reward:.3f}")
    
    # Reset
    obs, info = env.reset()
    
    # Test 3: Gradual pushing up
    print("\n3. Pushing legs to extend (thigh=-1, calf=1):")
    for i in range(10):
        # Try to extend legs
        action = np.array([
            0, -1, 1,  # FR
            0, -1, 1,  # FL
            0, -1, 1,  # RR
            0, -1, 1,  # RL
        ])
        obs, reward, _, _, info = env.step(action)
        if i % 2 == 0:
            print(f"   Step {i}: height={info['base_height']:.3f}m, reward={reward:.3f}")
    
    env.close()


def analyze_reward_landscape():
    """Analyze the reward landscape to understand local optima"""
    
    config = load_config()
    env = Go2Env(config)
    
    print("\n" + "=" * 70)
    print("REWARD LANDSCAPE ANALYSIS")
    print("=" * 70)
    
    # Multiple resets to see variance
    heights = []
    rewards = []
    orientations = []
    
    for _ in range(20):
        obs, info = env.reset()
        
        # Take a few random steps to settle
        for _ in range(5):
            action = np.zeros(12)
            obs, reward, _, _, info = env.step(action)
        
        heights.append(info['base_height'])
        rewards.append(reward)
        orientations.append(obs[24:27].copy())
    
    print(f"\nAfter reset + settle (20 trials):")
    print(f"  Height: {np.mean(heights):.3f} ± {np.std(heights):.3f}")
    print(f"  Reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"  θ_B z-component: {np.mean([o[2] for o in orientations]):.3f}")
    print(f"  (z=-1 is upright, z=+1 is upside down)")
    
    # Check: what's the maximum possible reward?
    print("\n\nTheoretical maximum rewards:")
    print("  R_h: 1.0 (when height >= 0.31m)")
    print("  R_g: 1.0 (when perfectly upright)")
    print("  R_ad: 1.0 (when action = previous action)")
    print("  R_v: 1.0 (when joints not moving)")
    print("  R_vb: 1.0 (when base not moving)")
    print("  Max total with current weights:")
    w = config['reward']['weights']
    max_reward = w['w1'] + w['w2'] + w['w6'] + w['w7'] + w['w8']
    print(f"    Without curriculum: {max_reward:.2f}")
    max_with_curriculum = max_reward + w['w3'] + w['w4'] + w['w5']
    print(f"    With curriculum: {max_with_curriculum:.2f}")
    
    env.close()


if __name__ == "__main__":
    test_rewards_in_different_states()
    test_action_effects()
    analyze_reward_landscape()
