"""
Test script to verify Go2 environment is working
"""

import sys
from pathlib import Path
import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from environment.go2_env import Go2Env


def load_config():
    """Load config from YAML"""
    config_path = Path(__file__).parent / "config" / "train_config.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_environment():
    """Test basic environment functionality"""
    
    print("=" * 60)
    print("Testing Go2 Self-Recovery Environment")
    print("=" * 60)
    
    # Load config
    print("\n1. Loading config...")
    config = load_config()
    
    # Add missing training config for testing
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
    
    print("   ✓ Config loaded")
    
    # Create environment
    print("\n2. Creating environment...")
    try:
        env = Go2Env(config, render_mode=None)
        print("   ✓ Environment created")
    except FileNotFoundError as e:
        print(f"   ✗ Error: {e}")
        print("\n   To fix: Download Go2 model from:")
        print("   https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2")
        return False
    except Exception as e:
        print(f"   ✗ Error creating environment: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test reset
    print("\n3. Testing reset...")
    try:
        obs, info = env.reset()
        print(f"   ✓ Reset successful")
        print(f"   - Observation shape: {obs.shape} (expected: (30,))")
        print(f"   - Observation dtype: {obs.dtype}")
        print(f"   - Base height: {info['base_height']:.3f}m")
        print(f"   - Obs range: [{obs.min():.2f}, {obs.max():.2f}]")
    except Exception as e:
        print(f"   ✗ Error during reset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test random actions
    print("\n4. Testing action execution...")
    try:
        n_test_steps = 10
        total_reward = 0
        
        for i in range(n_test_steps):
            # Random action
            action = env.action_space.sample()
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if i == 0:
                print(f"   ✓ First step successful")
                print(f"   - Action shape: {action.shape}")
                print(f"   - Reward: {reward:.4f}")
        
        print(f"   ✓ Completed {n_test_steps} steps")
        print(f"   - Average reward: {total_reward/n_test_steps:.4f}")
        print(f"   - Final height: {info['base_height']:.3f}m")
        
        # Check reward breakdown
        if 'reward_breakdown' in info:
            print(f"   - Reward breakdown:")
            for key, val in info['reward_breakdown'].items():
                if key != 'total':
                    print(f"     {key}: {val:.4f}")
        
    except Exception as e:
        print(f"   ✗ Error during step: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test episode rollout
    print("\n5. Testing full episode...")
    try:
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        max_steps = 100  # Short episode for testing
        
        for _ in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        print(f"   ✓ Episode completed")
        print(f"   - Length: {episode_length} steps")
        print(f"   - Total reward: {episode_reward:.4f}")
        print(f"   - Terminated: {terminated}, Truncated: {truncated}")
        
    except Exception as e:
        print(f"   ✗ Error during episode: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Cleanup
    env.close()
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Environment is working correctly.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Verify Go2 model parameters match your robot")
    print("  2. Tune PD controller gains (kp, kd)")
    print("  3. Adjust reward weights if needed")
    print("  4. Run training script")
    
    return True


if __name__ == "__main__":
    success = test_environment()
    sys.exit(0 if success else 1)
