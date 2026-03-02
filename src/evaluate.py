"""Evaluate trained model with visualization."""

import sys
from pathlib import Path
import yaml
import time
import mujoco.viewer
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO, SAC
from environment.go2_env import Go2Env


METHOD_ALIASES = {
    "ppo": "ppo",
    "sac": "sac",
    "cma-es": "cma-es",
    "cma_es": "cma-es",
    "cmaes": "cma-es",
    "cma": "cma",
}


METHOD_SPECS = {
    "ppo": {
        "logs_subdirs": ["ppo"],
        "type": "sb3",
        "loader": PPO,
        "candidates": [
            ("best", "best_model/best_model.zip"),
            ("final", "final_model.zip"),
            ("final", "final_model_.zip"),
            ("interrupted", "interrupted_model.zip"),
        ],
    },
    "sac": {
        "logs_subdirs": ["sac"],
        "type": "sb3",
        "loader": SAC,
        "candidates": [
            ("best", "best_model/best_model.zip"),
            ("final", "final_model.zip"),
            ("final", "final_model_.zip"),
            ("interrupted", "interrupted_model.zip"),
        ],
    },
    "cma-es": {
        "logs_subdirs": ["cma-es"],
        "type": "cma",
        "candidates": [
            ("best", "best_model/best_model.pt"),
            ("final", "final_model/final_model.pt"),
            ("best", "best_model.pt"),
            ("final", "final_model.pt"),
        ],
    },
    "cma": {
        "logs_subdirs": ["cma", "cma-es"],
        "type": "cma",
        "candidates": [
            ("best", "best_model/best_model.pt"),
            ("final", "final_model/final_model.pt"),
            ("best", "best_model.pt"),
            ("final", "final_model.pt"),
        ],
    },
}


class CMAPolicy(nn.Module):
    """Policy MLP compatible with models saved by CMA-ES training."""

    def __init__(self, obs_dim=30, act_dim=12, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, act_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        return self.net(obs)


class CMAModelWrapper:
    """Minimal wrapper to emulate Stable-Baselines `predict` interface."""

    def __init__(self, policy):
        self.policy = policy

    def predict(self, obs, deterministic=True):
        with torch.no_grad():
            action = self.policy(obs).cpu().numpy()
        return action, None


def ask_training_method():
    """Ask user which training method should be evaluated."""
    user_value = input(
        "Choose a training method model to evaluate:\n"
        " * ppo\n"
        " * sac\n"
        " * cma-es\n"
        " * cma\n"
    ).strip().lower()

    canonical = METHOD_ALIASES.get(user_value)
    if canonical is None:
        print(f"Unknown method '{user_value}', using 'ppo' as default.")
        return "ppo"
    return canonical


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


def find_latest_model(training_method):
    """Find the most recent trained model for selected training method."""
    method_spec = METHOD_SPECS[training_method]
    model_candidates = []

    for subdir in method_spec["logs_subdirs"]:
        logs_dir = project_root / "logs" / subdir
        if not logs_dir.exists():
            continue

        for run_dir in logs_dir.iterdir():
            if not run_dir.is_dir():
                continue

            for model_type, relative_model_path in method_spec["candidates"]:
                model_path = run_dir / relative_model_path
                if model_path.exists():
                    model_candidates.append(
                        (run_dir.stat().st_mtime, model_path, model_type, subdir)
                    )
                    break

    if not model_candidates:
        return None

    model_candidates.sort(reverse=True)
    _, model_path, model_type, source_subdir = model_candidates[0]
    return model_path, model_type, source_subdir


def load_trained_model(training_method, model_path, config):
    """Load trained model according to selected training method."""
    method_spec = METHOD_SPECS[training_method]

    if method_spec["type"] == "sb3":
        algorithm_class = method_spec["loader"]
        return algorithm_class.load(model_path)

    cma_config = config.get("cma_es", {})
    hidden_sizes = tuple(cma_config.get("hidden_sizes", [128, 128]))
    policy = CMAPolicy(obs_dim=30, act_dim=12, hidden_sizes=hidden_sizes)

    state_dict = torch.load(model_path, map_location="cpu")
    policy.load_state_dict(state_dict)
    policy.eval()

    return CMAModelWrapper(policy)


def evaluate():
    """Evaluate trained model with visualization"""
    training_method = ask_training_method()
    
    print("=" * 70)
    print("Model Evaluation - Go2 Self-Recovery")
    print("=" * 70)
    print(f"Training method: {training_method}")
    
    # Load config
    print("\n1. Loading configuration...")
    config = load_config()
    print("   ✓ Config loaded")
    
    # Find model
    print("\n2. Loading trained model...")
    model_info = find_latest_model(training_method)
    
    if model_info is None:
        print("   ✗ No trained model found!")
        print("\n   Please train a model first for the selected method.")
        return
    
    model_path, model_type, source_subdir = model_info
    print(f"   ✓ Found {model_type} model in logs/{source_subdir}: {model_path.parent.parent.name}")
    
    try:
        model = load_trained_model(training_method, model_path, config)
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
