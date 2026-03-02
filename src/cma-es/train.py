"""
Train CMA-ES agent for Go2 self-recovery
Based on paper: "Self-Recovery of Quadrupedal Robot Using DRL" (2024)
Using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

CMA-ES optimizes a neural network policy by evolving the weight vector.
Each candidate solution is a flat vector of NN parameters.
Fitness = mean cumulative reward over evaluation episodes.
"""

import sys
from pathlib import Path
import yaml
import numpy as np
from datetime import datetime
import os
import json
import time
import pickle

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cma
import torch
import torch.nn as nn

from environment.go2_env import Go2Env


# =============================================================================
# Neural Network Policy
# =============================================================================

class CMAPolicy(nn.Module):
    """
    Simple MLP policy for CMA-ES optimization.
    Maps observations to actions using a feedforward neural network.
    CMA-ES optimizes the flattened weight vector of this network.
    """

    def __init__(self, obs_dim=30, act_dim=12, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        prev_size = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, act_dim))
        layers.append(nn.Tanh())  # actions in [-1, 1]

        self.net = nn.Sequential(*layers)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def forward(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        return self.net(obs)

    def get_num_params(self):
        """Total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_flat_params(self):
        """Get all parameters as a single flat numpy array"""
        params = []
        for p in self.parameters():
            params.append(p.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_flat_params(self, flat_params):
        """Set all parameters from a single flat numpy array"""
        idx = 0
        for p in self.parameters():
            num = p.numel()
            p.data = torch.FloatTensor(
                flat_params[idx:idx + num].reshape(p.shape)
            )
            idx += num

    def act(self, obs):
        """Get action from observation (no gradient)"""
        with torch.no_grad():
            action = self.forward(obs)
        return action.numpy()


# =============================================================================
# Fitness Evaluation
# =============================================================================

def evaluate_policy(policy, config, n_episodes=5, max_steps=None, render=False):
    """
    Evaluate a policy on the environment and return mean cumulative reward.
    This is the fitness function for CMA-ES.

    Args:
        policy: CMAPolicy with weights set
        config: environment config dict
        n_episodes: number of episodes to average over
        max_steps: max steps per episode (from config if None)
        render: whether to render

    Returns:
        mean_reward: mean cumulative reward over episodes (NEGATED for CMA-ES minimization)
        info: dict with episode stats
    """
    if max_steps is None:
        max_steps = config['training'].get('max_episode_steps', 512)

    env = Go2Env(config, render_mode='human' if render else None)

    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0

        for step in range(max_steps):
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    env.close()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'episode_rewards': episode_rewards,
    }


def evaluate_candidate(flat_params, policy, config, n_episodes):
    """
    Evaluate a single candidate solution.
    Sets the policy weights and runs episodes.

    Returns:
        negative mean reward (CMA-ES minimizes, so we negate reward)
    """
    policy.set_flat_params(flat_params)
    mean_reward, _ = evaluate_policy(policy, config, n_episodes=n_episodes)
    return -mean_reward  # negate because CMA-ES minimizes


# =============================================================================
# Config & Logging
# =============================================================================

def load_config():
    """Load configuration from YAML"""
    config_path = project_root / "config" / "train_config.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(output_dir, generation, policy, es, best_reward, history):
    """Save training checkpoint"""
    checkpoint = {
        'generation': generation,
        'best_reward': best_reward,
        'best_params': policy.get_flat_params(),
        'cma_mean': es.mean.copy(),
        'cma_sigma': es.sigma,
        'history': history,
    }
    ckpt_path = output_dir / "checkpoints" / f"gen_{generation:05d}.pkl"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ckpt_path, 'wb') as f:
        pickle.dump(checkpoint, f)

    # Also save best model separately
    best_dir = output_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), best_dir / "best_model.pt")
    np.save(best_dir / "best_params.npy", policy.get_flat_params())


def load_checkpoint(checkpoint_path, policy):
    """Load a training checkpoint"""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    policy.set_flat_params(checkpoint['best_params'])
    return checkpoint


# =============================================================================
# Main Training Loop
# =============================================================================

def train():
    """Main CMA-ES training function"""

    print("=" * 70)
    print("CMA-ES Training - Go2 Self-Recovery")
    print("=" * 70)

    # Load config
    print("\n1. Loading configuration...")
    config = load_config()

    # CMA-ES specific config
    cma_config = config.get('cma_es', {})
    population_size = cma_config.get('population_size', 64)
    n_eval_episodes = cma_config.get('n_eval_episodes', 3)
    max_generations = cma_config.get('max_generations', 500)
    sigma0 = cma_config.get('sigma0', 0.5)
    hidden_sizes = cma_config.get('hidden_sizes', [128, 128])
    save_freq = cma_config.get('save_freq', 10)
    num_workers = cma_config.get('num_workers', 1)
    target_reward = cma_config.get('target_reward', None)

    # Ensure required config sections
    if 'training' not in config:
        config['training'] = {
            'drop_height': 0.3,
            'random_orientation': True,
            'random_joint_positions': True,
            'max_episode_steps': 512,
        }
    if 'simulation' not in config:
        config['simulation'] = {'n_substeps': 5}

    print("   ✓ Config loaded")

    # Create output directory
    timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    run_name = f"go2_{timestamp}"
    output_dir = project_root / "logs" / "cma-es" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"   ✓ Output directory: {output_dir}")

    # Save config
    with open(output_dir / "config.yml", 'w') as f:
        yaml.dump(config, f)

    # Create policy network
    print("\n2. Creating policy network...")
    policy = CMAPolicy(
        obs_dim=30,
        act_dim=12,
        hidden_sizes=hidden_sizes,
    )
    num_params = policy.get_num_params()
    print(f"   Architecture: {hidden_sizes}")
    print(f"   Total parameters: {num_params:,}")
    print(f"   Population size: {population_size}")
    print(f"   Sigma0: {sigma0}")
    print(f"   Eval episodes per candidate: {n_eval_episodes}")

    # Initialize CMA-ES
    print("\n3. Initializing CMA-ES optimizer...")
    x0 = policy.get_flat_params()

    cma_options = {
        'popsize': population_size,
        'maxiter': max_generations,
        'verb_disp': 1,
        'verb_log': 0,
        'seed': cma_config.get('seed', 42),
        'tolfun': 1e-11,
        'tolx': 1e-11,
    }

    # Bounded CMA-ES (optional, helps stability)
    bounds = cma_config.get('param_bounds', None)
    if bounds:
        cma_options['bounds'] = [bounds[0], bounds[1]]

    es = cma.CMAEvolutionStrategy(x0, sigma0, cma_options)
    print("   ✓ CMA-ES initialized")

    # Training loop
    print(f"\n4. Starting training...")
    print(f"   Max generations: {max_generations}")
    print(f"   Population per generation: {population_size}")
    print(f"   Total evaluations: ~{max_generations * population_size:,}")
    print("\n" + "=" * 70)
    print("Training in progress...")
    print("=" * 70 + "\n")

    history = {
        'generations': [],
        'best_rewards': [],
        'mean_rewards': [],
        'std_rewards': [],
        'sigma': [],
        'wall_time': [],
    }

    start_time = time.time()
    best_reward_ever = -np.inf
    best_params_ever = None

    try:
        generation = 0
        while not es.stop():
            generation += 1

            # Sample candidate solutions
            solutions = es.ask()

            # Evaluate each candidate
            gen_start = time.time()
            fitnesses = []

            if num_workers > 1:
                # Parallel evaluation using multiprocessing
                from multiprocessing import Pool
                with Pool(num_workers) as pool:
                    args = [
                        (sol, policy.__class__(30, 12, hidden_sizes), config, n_eval_episodes)
                        for sol in solutions
                    ]
                    # We need to pass params correctly for multiprocessing
                    results = []
                    for sol in solutions:
                        policy.set_flat_params(sol)
                        mean_rew, _ = evaluate_policy(policy, config, n_episodes=n_eval_episodes)
                        results.append(-mean_rew)
                    fitnesses = results
            else:
                # Sequential evaluation
                for i, sol in enumerate(solutions):
                    policy.set_flat_params(sol)
                    mean_reward, _ = evaluate_policy(policy, config, n_episodes=n_eval_episodes)
                    fitness = -mean_reward  # negate for minimization
                    fitnesses.append(fitness)

                    # Progress indicator
                    if (i + 1) % max(1, population_size // 4) == 0:
                        elapsed = time.time() - gen_start
                        print(f"   Gen {generation}: evaluated {i+1}/{population_size} "
                              f"candidates ({elapsed:.1f}s)")

            # Update CMA-ES distribution
            es.tell(solutions, fitnesses)

            # Statistics for this generation
            gen_best_fitness = min(fitnesses)
            gen_best_reward = -gen_best_fitness
            gen_mean_reward = -np.mean(fitnesses)
            gen_std_reward = np.std([-f for f in fitnesses])
            gen_time = time.time() - gen_start
            total_time = time.time() - start_time

            # Track best ever
            if gen_best_reward > best_reward_ever:
                best_reward_ever = gen_best_reward
                best_idx = np.argmin(fitnesses)
                best_params_ever = solutions[best_idx].copy()
                policy.set_flat_params(best_params_ever)

                # Save new best
                save_checkpoint(output_dir, generation, policy, es, best_reward_ever, history)
                improved = " ★ NEW BEST"
            else:
                improved = ""

            # Log
            print(f"\n{'='*70}")
            print(f"Generation {generation}/{max_generations} | "
                  f"Time: {total_time:.0f}s ({gen_time:.1f}s/gen)")
            print(f"{'='*70}")
            print(f"  Best reward (gen):   {gen_best_reward:8.3f}")
            print(f"  Mean reward (gen):   {gen_mean_reward:8.3f} ± {gen_std_reward:.3f}")
            print(f"  Best reward (ever):  {best_reward_ever:8.3f}{improved}")
            print(f"  CMA sigma:           {es.sigma:.6f}")
            print(f"{'='*70}\n")

            # Update history
            history['generations'].append(generation)
            history['best_rewards'].append(gen_best_reward)
            history['mean_rewards'].append(gen_mean_reward)
            history['std_rewards'].append(gen_std_reward)
            history['sigma'].append(es.sigma)
            history['wall_time'].append(total_time)

            # Periodic checkpoint
            if generation % save_freq == 0:
                policy.set_flat_params(best_params_ever if best_params_ever is not None else x0)
                save_checkpoint(output_dir, generation, policy, es, best_reward_ever, history)
                print(f"   ✓ Checkpoint saved at generation {generation}")

            # Early stopping on target reward
            if target_reward is not None and best_reward_ever >= target_reward:
                print(f"\n🎯 Target reward {target_reward} reached! Stopping.")
                break

        # Training finished
        print("\n" + "=" * 70)
        print("Training complete!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    # Save final results
    if best_params_ever is not None:
        policy.set_flat_params(best_params_ever)
    save_checkpoint(output_dir, generation, policy, es, best_reward_ever, history)

    # Save final model
    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), final_dir / "final_model.pt")
    np.save(final_dir / "final_params.npy", policy.get_flat_params())

    # Save training history as JSON
    history_json = {k: [float(v) for v in vals] if isinstance(vals, list) else vals
                    for k, vals in history.items()}
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history_json, f, indent=2)

    # Print summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Training Summary")
    print(f"{'='*70}")
    print(f"  Generations:        {generation}")
    print(f"  Best reward:        {best_reward_ever:.3f}")
    print(f"  Total time:         {total_time/60:.1f} minutes")
    print(f"  Total evaluations:  {generation * population_size}")
    print(f"  Output directory:   {output_dir}")
    print(f"{'='*70}\n")

    # Final evaluation of best policy
    print("Running final evaluation of best policy (5 episodes)...")
    policy.set_flat_params(best_params_ever)
    final_reward, final_info = evaluate_policy(policy, config, n_episodes=5)
    print(f"  Final eval reward: {final_reward:.3f} ± {final_info['std_reward']:.3f}")
    print(f"  Min/Max: {final_info['min_reward']:.3f} / {final_info['max_reward']:.3f}")
    print(f"  Mean episode length: {final_info['mean_length']:.1f}")


if __name__ == "__main__":
    train()
