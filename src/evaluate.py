"""Evaluate trained model with visualization."""

import sys
import argparse
from pathlib import Path
import yaml
import time
import importlib
import numpy as np
import mujoco
import mujoco.viewer
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO, SAC
from environment.go2_env import Go2Env
from src.models.qdecomp.networks import Actor as QDecompActor
from src.models.qdecomp_ppo.networks import MultiHeadPolicy as QDecompPPOPolicy


def _load_trpo_class():
    """Load TRPO class lazily to keep evaluate usable without sb3-contrib."""
    try:
        return importlib.import_module("sb3_contrib").TRPO
    except Exception:
        return None


TRPO = _load_trpo_class()


METHOD_ALIASES = {
    "ppo": "ppo",
    "trpo": "trpo",
    "sac": "sac",
    "cma-es": "cma-es",
    "cma_es": "cma-es",
    "cmaes": "cma-es",
    "cma": "cma",
    "qdecomp": "qdecomp",
    "q-decomp": "qdecomp",
    "qdecompsac": "qdecomp",
    "qdecompppo": "qdecomp_ppo",
    "qdecomp_ppo": "qdecomp_ppo",
    "q-decomp-ppo": "qdecomp_ppo",
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
    "trpo": {
        "logs_subdirs": ["trpo"],
        "type": "sb3",
        "loader": TRPO,
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
    "qdecomp": {
        "logs_subdirs": ["qdecomp"],
        "type": "qdecomp",
        "candidates": [
            ("best", "best_model.pt"),
            ("final", "final_model.pt"),
        ],
    },
    "qdecomp_ppo": {
        "logs_subdirs": ["qdecomp_ppo"],
        "type": "qdecomp_ppo",
        "candidates": [
            ("best", "best_model.pt"),
            ("final", "final_model.pt"),
        ],
    },
}


class CMAPolicy(nn.Module):
    """Policy MLP compatible with models saved by CMA-ES training."""

    def __init__(self, obs_dim=36, act_dim=12, hidden_sizes=(128, 128)):
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
        self.obs_dim = int(policy.net[0].in_features)

    def predict(self, obs, deterministic=True):
        with torch.no_grad():
            action = self.policy(obs).cpu().numpy()
        return action, None


class QDecompModelWrapper:
    """Wraps a QDecomp SAC actor to emulate the Stable-Baselines `predict` interface."""

    def __init__(self, actor: QDecompActor):
        self.actor = actor
        self.actor.eval()

    def predict(self, obs, deterministic=True):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(obs_t)
            else:
                action, _ = self.actor.sample(obs_t)
        return action.squeeze(0).cpu().numpy(), None


class QDecompPPOModelWrapper:
    """Wraps a QDecomp PPO policy to emulate the Stable-Baselines `predict` interface."""

    def __init__(self, policy: QDecompPPOPolicy):
        self.policy = policy
        self.policy.eval()

    def predict(self, obs, deterministic=True):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.policy.predict(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy(), None


def resolve_training_method(training_method):
    """Resolve model method from CLI arg using known aliases."""
    canonical = METHOD_ALIASES.get((training_method or "ppo").strip().lower())
    if canonical is None:
        print(f"Unknown method '{training_method}', using 'ppo' as default.")
        return "ppo"
    return canonical


def adapt_observation_for_model(obs: np.ndarray, expected_dim) -> np.ndarray:
    """Trim or pad observation to match model input size."""
    if expected_dim is None or expected_dim <= 0:
        return obs
    if obs.shape[0] == expected_dim:
        return obs
    if obs.shape[0] > expected_dim:
        return obs[:expected_dim]

    padded = np.zeros(expected_dim, dtype=obs.dtype)
    padded[:obs.shape[0]] = obs
    return padded


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
        if algorithm_class is None:
            raise ImportError(
                "TRPO support requires sb3-contrib. Install it with: pip install sb3-contrib"
            )
        return algorithm_class.load(model_path)

    if method_spec["type"] == "qdecomp":
        checkpoint = torch.load(model_path, map_location="cpu")
        qd_cfg = config.get("qdecomp", {})
        hidden = qd_cfg.get("hidden_sizes", [256, 256])
        # Infer obs_dim from saved weights to stay compatible with any env version
        obs_dim = checkpoint["actor"]["net.0.weight"].shape[1]
        actor = QDecompActor(obs_dim=obs_dim, action_dim=12, hidden_sizes=hidden)
        actor.load_state_dict(checkpoint["actor"])
        return QDecompModelWrapper(actor)

    if method_spec["type"] == "qdecomp_ppo":
        checkpoint = torch.load(model_path, map_location="cpu")
        qd_cfg = config.get("qdecomp_ppo", {})
        hidden = qd_cfg.get("hidden_sizes", [256, 256])
        # Infer obs_dim from saved trunk weights
        obs_dim = checkpoint["policy"]["trunk.0.weight"].shape[1]
        policy = QDecompPPOPolicy(obs_dim=obs_dim, action_dim=12, hidden_sizes=hidden)
        policy.load_state_dict(checkpoint["policy"])
        return QDecompPPOModelWrapper(policy)

    cma_config = config.get("cma_es", {})
    hidden_sizes = tuple(cma_config.get("hidden_sizes", [128, 128]))
    state_dict = torch.load(model_path, map_location="cpu")

    # Infer observation dimension from checkpoint for backward compatibility.
    # Old models may have obs_dim=30 while new models use obs_dim=36.
    first_linear_key = None
    for key in state_dict.keys():
        if key.endswith("0.weight"):
            first_linear_key = key
            break
    inferred_obs_dim = int(state_dict[first_linear_key].shape[1]) if first_linear_key else 36

    policy = CMAPolicy(obs_dim=inferred_obs_dim, act_dim=12, hidden_sizes=hidden_sizes)
    policy.load_state_dict(state_dict)
    policy.eval()

    return CMAModelWrapper(policy)


def place_robot_for_terrain(env, terrain_mode, level=0):
    """Place robot on selected terrain family and roughness level in scene.xml."""
    if terrain_mode in {"flat_inclined", "rough"}:
        level = int(np.clip(level, 0, 8))
    else:
        level = int(np.clip(level, 0, 9))

    x_step = 3.0 if terrain_mode == "rough" else 3.25
    x_pos = x_step * level
    incline_deg = float(level + 1)
    incline_rad = np.deg2rad(incline_deg)
    incline_center_z = -0.05 + 1.5 * np.sin(incline_rad)

    # Harder terrains benefit from tiny spawn jitter and extra lift to avoid starts
    # with immediate foot wedging/contact locking.
    difficult_terrains = {"rocky", "rocky_inclined", "rough"}
    if terrain_mode in difficult_terrains:
        x_pos += np.random.uniform(-0.35, 0.35)
        y_jitter = np.random.uniform(-0.25, 0.25)
    else:
        y_jitter = 0.0

    spawn_lift_map = {
        "flat": 0.0,
        "flat_inclined": 0.02,
        "rocky": 0.06,
        "rocky_inclined": 0.08,
        "rough": 0.08,
        "default": 0.0,
        "inclined": 0.08,
        "smooth_inclined": 0.02,
    }
    spawn_lift = float(spawn_lift_map.get(terrain_mode, 0.0))

    spawn_map = {
        "flat": {"pos": [0.0, 0.0, 0.12 + spawn_lift], "pitch_deg": 0.0},
        "flat_inclined": {"pos": [x_pos, 12.0 + y_jitter, 0.24 + incline_center_z + spawn_lift], "pitch_deg": -incline_deg},
        "rocky": {"pos": [x_pos, 3.0 + y_jitter, 0.40 + spawn_lift], "pitch_deg": 0.0},
        "rocky_inclined": {"pos": [x_pos, 6.0 + y_jitter, 0.24 + incline_center_z + spawn_lift], "pitch_deg": -incline_deg},
        "rough": {"pos": [x_pos, 9.0 + y_jitter, 0.24 + incline_center_z + spawn_lift], "pitch_deg": -incline_deg},
        # Legacy aliases
        "default": {"pos": [0.0, 0.0, 0.12 + spawn_lift], "pitch_deg": 0.0},
        "inclined": {"pos": [x_pos, 6.0 + y_jitter, 0.24 + incline_center_z + spawn_lift], "pitch_deg": -incline_deg},
        "smooth_inclined": {"pos": [x_pos, 12.0 + y_jitter, 0.24 + incline_center_z + spawn_lift], "pitch_deg": -incline_deg},
    }

    spawn = spawn_map.get(terrain_mode, spawn_map["flat"])

    env.data.qpos[0:3] = spawn["pos"]

    pitch = np.deg2rad(spawn["pitch_deg"])
    yaw = np.random.uniform(-np.pi, np.pi)
    quat = env._euler_to_quat([np.pi, pitch, yaw])
    env.data.qpos[3:7] = quat

    tucked_config = np.array([
        0.0, 1.8, -2.4,
        0.0, 1.8, -2.4,
        0.0, 1.8, -2.4,
        0.0, 1.8, -2.4,
    ])
    for i, addr in enumerate(env.joint_qpos_addr):
        env.data.qpos[addr] = tucked_config[i]

    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    # Let contacts settle to avoid floating starts on uneven terrain.
    for _ in range(60):
        env.data.ctrl[:] = 0
        mujoco.mj_step(env.model, env.data)


def place_robot_with_retries(env, terrain_mode, level=0, max_retries=10):
    """Try multiple placements and keep the first contact-valid spawn."""
    for attempt in range(max_retries):
        place_robot_for_terrain(env, terrain_mode, level=level)
        # Use env validity check when available to reject wedged initial states.
        if not hasattr(env, "_is_valid_spawn_state"):
            return True, attempt
        if env._is_valid_spawn_state():
            return True, attempt
    return False, max_retries


def place_robot_for_curriculum_phase(env, phase_idx=0):
    """Place robot using a specific dynamic curriculum phase index."""
    terrains = env.terrain_curriculum.get_all_terrains()
    phase_idx = int(np.clip(phase_idx, 0, len(terrains) - 1))
    terrain_config = terrains[phase_idx]
    env._place_robot_for_terrain(terrain_config)
    return phase_idx, terrain_config


def place_robot_for_flat_inclined_level(env, level=0):
    """Place robot on smooth flat inclined row in scene.xml (1..9 deg)."""
    level = int(np.clip(level, 0, 8))
    incline_deg = level + 1
    terrain_config = {
        "name": f"smooth_flat_incline_{incline_deg}",
        "roughness_level": 0,
        "pitch_deg": -float(incline_deg),
        "incline_deg": float(incline_deg),
    }
    place_robot_for_terrain(env, "flat_inclined", level=level)
    return level, terrain_config


def update_viewer_camera(viewer, env):
    """Keep camera centered around robot base position."""
    base_pos = env.data.qpos[0:3]
    viewer.cam.lookat[0] = float(base_pos[0])
    viewer.cam.lookat[1] = float(base_pos[1])
    viewer.cam.lookat[2] = float(base_pos[2])


def _foot_contact_label(feet_contacts):
    """Return human-readable per-foot contact labels."""
    labels = ["FR", "FL", "RR", "RL"]
    values = []
    for i, label in enumerate(labels):
        in_contact = bool(feet_contacts[i]) if i < len(feet_contacts) else False
        values.append(f"{label}: {'ON' if in_contact else 'OFF'}")
    return values


def print_contact_terminal(info, step):
    """Print per-foot contact states and total contact count in terminal."""
    feet_contacts = info.get("feet_contacts", [False, False, False, False])
    contact_count = int(info.get("feet_contact_count", int(sum(feet_contacts))))
    rb = info.get("reward_breakdown", {})
    touchdowns_step = int(rb.get("foot_contact_touchdowns", 0.0))
    touchdowns_total = int(rb.get("foot_contact_touchdowns_total", 0.0))
    labels = _foot_contact_label(feet_contacts)
    line = (
        f"  Contacts | Step {step:4d} | Count: {contact_count} | "
        f"Touch: {touchdowns_step} (tot {touchdowns_total}) | "
        f"{labels[0]} | {labels[1]} | {labels[2]} | {labels[3]}"
    )
    print(f"\r{line}", end="", flush=True)


def evaluate(training_method=None, terrain_mode="default", level=0, random_terrain=False):
    """Evaluate trained model with visualization"""
    training_method = resolve_training_method(training_method)
    valid_terrains = {
        "flat", "flat_inclined", "rocky", "rocky_inclined", "rough", "curriculum",
        # Legacy aliases
        "default", "inclined", "smooth_inclined",
    }
    if terrain_mode not in valid_terrains:
        print(f"Unknown terrain '{terrain_mode}', using 'flat'.")
        terrain_mode = "flat"

    # Normalize legacy names to the new terrain taxonomy.
    terrain_aliases = {
        "default": "flat",
        "inclined": "rocky_inclined",
        "smooth_inclined": "flat_inclined",
    }
    terrain_mode = terrain_aliases.get(terrain_mode, terrain_mode)

    level = int(level)
    
    print("=" * 70)
    print("Model Evaluation - Go2 Self-Recovery")
    print("=" * 70)
    print(f"Training method: {training_method}")
    
    # Load config
    print("\n1. Loading configuration...")
    config = load_config()
    config.setdefault("terrain_curriculum", {})
    config["terrain_curriculum"]["enabled"] = False
    config["terrain_curriculum"]["default_terrain"] = "flat"
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

    expected_obs_dim = None
    if hasattr(model, "observation_space") and model.observation_space is not None:
        expected_obs_dim = int(model.observation_space.shape[0])
    elif hasattr(model, "obs_dim"):
        expected_obs_dim = int(model.obs_dim)

    if expected_obs_dim is not None:
        print(f"   Model observation dim: {expected_obs_dim}")
    
    # Create environment
    # eval_mode=True forces the real self-recovery task: robot always starts
    # belly-UP (supine), ignoring the prone training curriculum.
    print("\n3. Creating environment...")
    env = Go2Env(config, render_mode=None, eval_mode=True)
    print("   ✓ Environment created (eval_mode: belly-up SR start)")
    
    print("\n4. Starting evaluation...")
    print(f"   Terrain mode: {terrain_mode}")
    if random_terrain:
        print("   Terrain sampling: RANDOM per episode")
        print("   Modes sampled: flat | flat_inclined | rocky | rocky_inclined | rough")
    else:
        if terrain_mode in {"flat_inclined", "rough"}:
            print(f"   Flat inclined level (--level): {int(np.clip(level, 0, 8))} (range: 0-8)")
            if terrain_mode == "flat_inclined":
                print("   Mapping: level 0..8 -> incline 1°..9° (totally smooth)")
            else:
                print("   Mapping: level 0..8 -> incline 1°..9° (rough, non-rocky)")
        elif terrain_mode == "rocky_inclined":
            print(f"   Rocky inclined level (--level): {int(np.clip(level, 0, 9))} (range: 0-9)")
        elif terrain_mode == "rocky":
            print(f"   Rocky level (--level): {int(np.clip(level, 0, 9))} (range: 0-9)")
        elif terrain_mode == "flat":
            print("   Flat terrain: totally smooth at 0°")
        elif terrain_mode == "curriculum":
            max_phase = env.terrain_curriculum.get_terrain_count() - 1
            print(f"   Curriculum phase (--level): {int(np.clip(level, 0, max_phase))} (range: 0-{max_phase})")
        else:
            print(f"   Terrain level: {int(np.clip(level, 0, 9))}")
    print("\nControls:")
    print("  - Double-click + drag: Rotate camera")
    print("  - Right-click + drag: Pan camera")
    print("  - Scroll: Zoom in/out")
    print("  - Close window or Ctrl+C to stop\n")
    
    # Launch viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        try:
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 135

            episode = 1
            total_episodes = 5  # Run 5 episodes
            
            while episode <= total_episodes and viewer.is_running():
                print(f"\n{'='*50}")
                print(f"Episode {episode}/{total_episodes}")
                print(f"{'='*50}")

                episode_terrain = terrain_mode
                episode_level = int(level)
                if random_terrain:
                    episode_terrain = str(np.random.choice([
                        "flat", "flat_inclined", "rocky", "rocky_inclined", "rough"
                    ]))
                    if episode_terrain in {"flat_inclined", "rough"}:
                        episode_level = int(np.random.randint(0, 9))
                    elif episode_terrain in {"rocky", "rocky_inclined"}:
                        episode_level = int(np.random.randint(0, 10))
                    else:
                        episode_level = 0
                    print(f"Random terrain draw: {episode_terrain} (level={episode_level})")
                
                # Reset environment
                obs, info = env.reset()
                if episode_terrain == "flat_inclined":
                    level_idx, terrain_cfg = place_robot_for_flat_inclined_level(env, level=episode_level)
                    obs = env._get_observation()
                    info = env._get_info()
                    print(
                        f"Flat inclined level {level_idx}: {terrain_cfg['name']} "
                        f"(roughness={terrain_cfg['roughness_level']}, pitch={terrain_cfg['pitch_deg']:.1f}°)"
                    )
                elif episode_terrain == "rough":
                    level_idx = int(np.clip(episode_level, 0, 8))
                    place_robot_for_terrain(env, "rough", level=level_idx)
                    obs = env._get_observation()
                    info = env._get_info()
                    print(
                        f"Rough level {level_idx}: "
                        f"(non-rocky irregular, pitch={-(level_idx + 1):.1f}°)"
                    )
                elif episode_terrain == "curriculum":
                    phase_idx, terrain_cfg = place_robot_for_curriculum_phase(env, phase_idx=episode_level)
                    obs = env._get_observation()
                    info = env._get_info()
                    print(
                        f"Curriculum phase {phase_idx}: {terrain_cfg['name']} "
                        f"(roughness={terrain_cfg['roughness_level']}, pitch={terrain_cfg['pitch_deg']:.1f}°)"
                    )
                elif episode_terrain != "flat" or episode_level != 0:
                    spawn_ok, spawn_attempt = place_robot_with_retries(
                        env,
                        episode_terrain,
                        level=int(np.clip(episode_level, 0, 9)),
                        max_retries=12,
                    )
                    obs = env._get_observation()
                    info = env._get_info()
                    if not spawn_ok:
                        print("  Warning: spawn remained contact-heavy after retries.")
                    elif spawn_attempt > 0:
                        print(f"  Spawn retries used: {spawn_attempt}")

                update_viewer_camera(viewer, env)
                print_contact_terminal(info, step=0)
                print(f"Initial height: {info['base_height']:.3f}m")
                
                episode_reward = 0
                step = 0
                done = False
                
                # Run episode with trained policy
                while not done and viewer.is_running():
                    # Get action from trained policy
                    model_obs = adapt_observation_for_model(obs, expected_obs_dim)
                    action, _states = model.predict(model_obs, deterministic=True)
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    step += 1
                    
                    # Sync viewer
                    update_viewer_camera(viewer, env)
                    print_contact_terminal(info, step=step)
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

                # End the in-place contact status line before summary prints.
                print()
                
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
    parser = argparse.ArgumentParser(description="Evaluate latest trained model")
    parser.add_argument(
        "--model",
        type=str,
        default="ppo",
        help="Model type: ppo | trpo | sac | cma-es | cma | qdecomp | qdecomp_ppo",
    )
    parser.add_argument(
        "--terrain",
        type=str,
        default="flat",
        choices=["flat", "flat_inclined", "rocky", "rocky_inclined", "rough", "curriculum", "default", "inclined", "smooth_inclined"],
        help="Spawn terrain: flat | flat_inclined | rocky | rocky_inclined | rough | curriculum",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="Terrain level index. flat_inclined/rough: 0-8. rocky/rocky_inclined: 0-9. curriculum: phase index.",
    )
    parser.add_argument(
        "--inclined",
        action="store_true",
        help="Deprecated alias for --terrain inclined",
    )
    parser.add_argument(
        "--random-terrain",
        action="store_true",
        help="Sample random terrain (and valid level) at each episode.",
    )
    args = parser.parse_args()

    terrain = args.terrain
    if args.inclined and terrain in {"default", "flat"}:
        terrain = "rocky_inclined"

    evaluate(
        training_method=args.model,
        terrain_mode=terrain,
        level=args.level,
        random_terrain=args.random_terrain,
    )
