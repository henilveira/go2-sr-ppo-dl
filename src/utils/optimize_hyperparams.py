"""
Hyperparameter Optimization using Optuna
Optimizes algorithm hyperparameters for Go2 self-recovery
"""

import sys
import csv
from pathlib import Path
import yaml
import numpy as np
from datetime import datetime, timedelta
import time
import optuna
from functools import partial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from environment.go2_env import Go2Env


ACTIVATION_FNS = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "elu": torch.nn.ELU,
}


class TrainingProgressCallback(BaseCallback):
    """Shows training progress in real-time"""
    
    def __init__(self, trial_number, total_timesteps, update_interval=10000):
        super().__init__()
        self.trial_number = trial_number
        self.total_timesteps = total_timesteps
        self.update_interval = update_interval
        self.last_update = 0
        self.start_time = None
    
    def _on_training_start(self):
        self.start_time = time.time()
    
    def _on_step(self):
        if self.num_timesteps - self.last_update >= self.update_interval:
            elapsed = time.time() - self.start_time
            progress = self.num_timesteps / self.total_timesteps
            steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
            eta_seconds = (self.total_timesteps - self.num_timesteps) / steps_per_sec if steps_per_sec > 0 else 0
            eta = timedelta(seconds=int(eta_seconds))
            
            # Get recent episode info if available
            ep_info = ""
            if len(self.model.ep_info_buffer) > 0:
                ep_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                ep_info = f" | Reward: {ep_reward:.1f}"
            
            print(f"    [{progress*100:5.1f}%] {self.num_timesteps:7,}/{self.total_timesteps:,} steps | "
                  f"{steps_per_sec:.0f} SPS | ETA: {str(eta).split('.')[0]}{ep_info}", flush=True)
            
            self.last_update = self.num_timesteps
        return True


class ProgressCallback:
    """Callback to show progress during optimization"""
    
    def __init__(self, n_trials):
        self.n_trials = n_trials
        self.start_time = time.time()
        self.trial_times = []
    
    def __call__(self, study, trial):
        trial_time = (time.time() - self.start_time) - sum(self.trial_times)
        self.trial_times.append(trial_time)
        
        n_completed = len(self.trial_times)
        avg_time = np.mean(self.trial_times)
        remaining = self.n_trials - n_completed
        eta_seconds = remaining * avg_time
        eta = timedelta(seconds=int(eta_seconds))
        
        elapsed = timedelta(seconds=int(time.time() - self.start_time))
        progress = n_completed / self.n_trials * 100
        
        print(f"\n{'='*70}")
        print(f"📊 PROGRESS: {n_completed}/{self.n_trials} trials ({progress:.1f}%)")
        print(f"⏱️  Elapsed: {elapsed} | ETA: {eta} | Avg: {avg_time/60:.1f} min/trial")
        print(f"🏆 Best reward: {study.best_value:.2f} (trial #{study.best_trial.number + 1})")
        print(f"{'='*70}\n")


def load_config():
    """Load base configuration"""
    config_path = project_root / "config" / "train_config.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def make_env(config, rank=0):
    """Factory function to create environment"""
    def _init():
        if 'training' not in config:
            config['training'] = {
                'drop_height': 0.3,
                'random_orientation': True,
                'random_joint_positions': True,
                'max_episode_steps': 512
            }
        
        if 'simulation' not in config:
            config['simulation'] = {
                'n_substeps': 10
            }
        
        env = Go2Env(config, render_mode=None)
        env = Monitor(env)
        return env
    return _init


def calculate_total_timesteps(config, model_name, n_envs, timesteps_override=None):
    """Compute model-specific timesteps for HPO."""
    model_name = model_name.lower()

    if timesteps_override is not None:
        return int(timesteps_override)

    # Targets requested for HPO runtime budget
    default_targets = {
        'ppo': 2_500_400,   # ~2.5M
        'sac': 3_686_400,   # ~3.7M
    }

    target = int(default_targets.get(model_name, 2_500_400))

    if model_name == 'ppo':
        # PPO should use a multiple of n_steps * n_envs for clean rollouts.
        ppo_n_steps = int(config.get('ppo', {}).get('n_steps', 1024))
        rollout_block = max(ppo_n_steps * max(n_envs, 1), 1)
        return max(rollout_block, round(target / rollout_block) * rollout_block)

    return target


def get_model_param_names(model_name):
    """Return Optuna parameter names for the selected model."""
    if model_name.lower() == 'ppo':
        return [
            'learning_rate',
            'batch_size',
            'gamma',
            'gae_lambda',
            'clip_range',
            'ent_coef',
            'vf_coef',
            'max_grad_norm',
            'n_epochs',
        ]

    if model_name.lower() == 'sac':
        return [
            'learning_rate',
            'batch_size',
            'tau',
            'gamma',
            'train_freq',
            'gradient_steps',
            'learning_starts',
        ]

    return []


def save_trial_history_csv(study, output_dir, model_name, timesteps_per_trial):
    """Save per-trial optimization history as CSV."""
    csv_path = output_dir / 'trial_history.csv'
    param_names = get_model_param_names(model_name)
    fieldnames = [
        'trial_number',
        'state',
        'value',
        'timesteps_per_trial',
        'cumulative_timesteps',
        'duration_seconds',
    ] + param_names

    completed_count = 0
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                completed_count += 1

            duration_seconds = None
            if trial.datetime_start is not None and trial.datetime_complete is not None:
                duration_seconds = (trial.datetime_complete - trial.datetime_start).total_seconds()

            row = {
                'trial_number': trial.number,
                'state': trial.state.name,
                'value': trial.value,
                'timesteps_per_trial': timesteps_per_trial,
                'cumulative_timesteps': completed_count * timesteps_per_trial if trial.state == optuna.trial.TrialState.COMPLETE else '',
                'duration_seconds': duration_seconds,
            }
            for param_name in param_names:
                row[param_name] = trial.params.get(param_name, '')
            writer.writerow(row)

    return csv_path


def _plot_parameter_axis(axis, x_values, param_values, param_name):
    """Plot one parameter series, supporting numeric and categorical values."""
    numeric = True
    converted_values = []
    for value in param_values:
        if isinstance(value, (int, float, np.integer, np.floating)):
            converted_values.append(float(value))
        else:
            numeric = False
            break

    if numeric:
        axis.plot(x_values, converted_values, marker='o', linewidth=1.5)
        axis.set_ylabel(param_name)
        return

    categories = list(dict.fromkeys(param_values))
    mapping = {category: idx for idx, category in enumerate(categories)}
    encoded_values = [mapping[value] for value in param_values]
    axis.plot(x_values, encoded_values, marker='o', linewidth=1.5)
    axis.set_yticks(range(len(categories)))
    axis.set_yticklabels([str(category) for category in categories])
    axis.set_ylabel(param_name)


def save_optimization_plots(study, output_dir, model_name, timesteps_per_trial):
    """Save reward and parameter evolution plots for completed trials."""
    if plt is None:
        print("⚠ matplotlib not installed; skipping optimization plots.")
        return []

    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        return []

    x_values = np.arange(1, len(completed_trials) + 1) * timesteps_per_trial / 1_000_000
    rewards = np.array([trial.value for trial in completed_trials], dtype=float)
    best_so_far = np.maximum.accumulate(rewards)
    param_names = get_model_param_names(model_name)

    reward_plot_path = output_dir / 'reward_evolution.png'
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, rewards, marker='o', linewidth=1.5, label='Trial reward')
    plt.plot(x_values, best_so_far, linewidth=2.0, label='Best so far')
    plt.xlabel('Cumulative optimization timesteps (millions)')
    plt.ylabel('Mean reward')
    plt.title(f'{model_name.upper()} optimization reward evolution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(reward_plot_path, dpi=200)
    plt.close()

    saved_paths = [reward_plot_path]

    if param_names:
        fig, axes = plt.subplots(len(param_names), 1, figsize=(10, max(3 * len(param_names), 6)), sharex=True)
        if len(param_names) == 1:
            axes = [axes]

        for axis, param_name in zip(axes, param_names):
            param_values = [trial.params[param_name] for trial in completed_trials]
            _plot_parameter_axis(axis, x_values, param_values, param_name)
            axis.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Cumulative optimization timesteps (millions)')
        fig.suptitle(f'{model_name.upper()} hyperparameter evolution', y=0.995)
        fig.tight_layout()
        params_plot_path = output_dir / 'hyperparameter_evolution.png'
        fig.savefig(params_plot_path, dpi=200)
        plt.close(fig)
        saved_paths.append(params_plot_path)

    return saved_paths


def apply_trial_hyperparameters(config, trial, model_name):
    """Apply model-specific Optuna suggestions to config."""
    model_name = model_name.lower()

    trial_params = {}

    if model_name == 'ppo':
        ppo_params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'gamma': trial.suggest_float('gamma', 0.95, 0.999),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
            'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
            'ent_coef': trial.suggest_float('ent_coef', 1e-6, 1e-2, log=True),
            'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
            'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 1.0),
            'n_epochs': trial.suggest_categorical('n_epochs', [5, 10, 15, 20]),
        }
        config['ppo'].update(ppo_params)
        trial_params['ppo'] = ppo_params

    if model_name == 'sac':
        sac_params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
            'tau': trial.suggest_float('tau', 0.001, 0.02, log=True),
            'gamma': trial.suggest_float('gamma', 0.95, 0.999),
            'train_freq': trial.suggest_categorical('train_freq', [1, 2, 4, 8]),
            'gradient_steps': trial.suggest_categorical('gradient_steps', [1, 2, 4, 8]),
            'learning_starts': trial.suggest_categorical('learning_starts', [5000, 10000, 20000, 50000]),
        }
        config['sac'].update(sac_params)
        trial_params['sac'] = sac_params

    return trial_params


def print_trial_hyperparameters(model_name, trial_params):
    """Print model-specific hyperparameters under test."""
    print(f"\nTesting hyperparameters:")
    print(f"  model:     {model_name.upper()}")

    if model_name.lower() == 'ppo':
        ppo_params = trial_params['ppo']
        print(f"  lr:        {ppo_params['learning_rate']:.2e}")
        print(f"  batch:     {ppo_params['batch_size']}")
        print(f"  gamma:     {ppo_params['gamma']:.4f}")
        print(f"  gae:       {ppo_params['gae_lambda']:.4f}")
        print(f"  clip:      {ppo_params['clip_range']:.3f}")
        print(f"  ent:       {ppo_params['ent_coef']:.2e}")
        print(f"  vf:        {ppo_params['vf_coef']:.3f}")
        print(f"  gradnorm:  {ppo_params['max_grad_norm']:.3f}")
        print(f"  epochs:    {ppo_params['n_epochs']}")

    if model_name.lower() == 'sac':
        sac_params = trial_params['sac']
        print(f"  lr:        {sac_params['learning_rate']:.2e}")
        print(f"  batch:     {sac_params['batch_size']}")
        print(f"  tau:       {sac_params['tau']:.4f}")
        print(f"  gamma:     {sac_params['gamma']:.4f}")
        print(f"  freq:      {sac_params['train_freq']}")
        print(f"  grad:      {sac_params['gradient_steps']}")
        print(f"  starts:    {sac_params['learning_starts']}")


def build_best_params_payload(model_name, best_trial, n_trials):
    """Build YAML payload with common and model-specific best params."""
    payload = {
        'optimization_info': {
            'model': model_name.lower(),
            'best_reward': best_trial.value,
            'trial_number': best_trial.number,
            'n_trials': n_trials,
        },
    }

    if model_name.lower() == 'ppo':
        payload['ppo'] = {
            'learning_rate': best_trial.params['learning_rate'],
            'batch_size': best_trial.params['batch_size'],
            'gamma': best_trial.params['gamma'],
            'gae_lambda': best_trial.params['gae_lambda'],
            'clip_range': best_trial.params['clip_range'],
            'ent_coef': best_trial.params['ent_coef'],
            'vf_coef': best_trial.params['vf_coef'],
            'max_grad_norm': best_trial.params['max_grad_norm'],
            'n_epochs': best_trial.params['n_epochs'],
        }

    if model_name.lower() == 'sac':
        payload['sac'] = {
            'learning_rate': best_trial.params['learning_rate'],
            'batch_size': best_trial.params['batch_size'],
            'tau': best_trial.params['tau'],
            'gamma': best_trial.params['gamma'],
            'train_freq': best_trial.params['train_freq'],
            'gradient_steps': best_trial.params['gradient_steps'],
            'learning_starts': best_trial.params['learning_starts'],
        }

    return payload


def get_top_trials_columns(model_name):
    """Return dataframe columns for top-trials summary."""
    columns = ['number', 'value']

    if model_name.lower() == 'ppo':
        columns.extend([
            'params_learning_rate',
            'params_batch_size',
            'params_gamma',
            'params_gae_lambda',
            'params_clip_range',
            'params_ent_coef',
            'params_vf_coef',
            'params_max_grad_norm',
            'params_n_epochs',
        ])

    if model_name.lower() == 'sac':
        columns.extend([
            'params_learning_rate',
            'params_batch_size',
            'params_tau',
            'params_gamma',
            'params_train_freq',
            'params_gradient_steps',
            'params_learning_starts',
        ])

    return columns


def create_model(model_name, config, env):
    """Create SB3 model based on selected algorithm."""
    model_name = model_name.lower()

    if model_name == "ppo":
        ppo_config = config['ppo']
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=ppo_config['learning_rate'],
            n_steps=ppo_config['n_steps'],
            batch_size=ppo_config['batch_size'],
            n_epochs=ppo_config['n_epochs'],
            gamma=ppo_config['gamma'],
            gae_lambda=ppo_config['gae_lambda'],
            clip_range=ppo_config['clip_range'],
            ent_coef=ppo_config['ent_coef'],
            vf_coef=ppo_config['vf_coef'],
            max_grad_norm=ppo_config['max_grad_norm'],
            policy_kwargs={
                'net_arch': ppo_config['policy_kwargs']['net_arch'],
                'activation_fn': torch.nn.ReLU,
            },
            verbose=0,
        )

    if model_name == "sac":
        sac_config = config['sac']
        act_fn_name = sac_config['policy_kwargs'].get('activation_fn', 'relu')
        activation_fn = ACTIVATION_FNS.get(str(act_fn_name).lower(), torch.nn.ReLU)

        ent_coef = sac_config['ent_coef']
        if not (isinstance(ent_coef, str) and ent_coef == "auto"):
            ent_coef = float(ent_coef)

        target_entropy = sac_config.get('target_entropy', 'auto')
        if not (isinstance(target_entropy, str) and target_entropy == "auto"):
            target_entropy = float(target_entropy)

        return SAC(
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
            verbose=0,
        )

    raise ValueError(f"Unsupported model '{model_name}'. Use 'ppo' or 'sac'.")


def objective(trial, model_name, timesteps_override=None):
    """
    Optuna objective function - trains and evaluates model with suggested hyperparameters
    """
    
    trial_start_time = time.time()
    
    n_total_trials = trial.study.user_attrs.get('n_trials', '?')
    completed_trials = trial.study.get_trials(
        deepcopy=False,
        states=(optuna.trial.TrialState.COMPLETE,),
    )
    best_so_far = None
    if completed_trials:
        # Safe in parallel mode: only read best_value when at least one trial completed.
        best_so_far = trial.study.best_value
    
    print(f"\n{'='*70}")
    if best_so_far is not None:
        print(f"Trial {trial.number + 1}/{n_total_trials} | Best so far: {best_so_far:.2f}")
    else:
        print(f"Trial {trial.number + 1}/{n_total_trials}")
    print(f"{'='*70}")
    
    # Load base config
    config = load_config()
    model_name = model_name.lower()
    
    # ============================================================
    # HYPERPARAMETERS TO OPTIMIZE
    # ============================================================
    
    trial_params = apply_trial_hyperparameters(config, trial, model_name)
    print_trial_hyperparameters(model_name, trial_params)
    
    # ============================================================
    # TRAIN MODEL (shorter training for HPO)
    # ============================================================
    
    # In parallel Optuna runs, avoid nested multiprocessing (SubprocVecEnv)
    # because MuJoCo + threaded trials can become unstable on macOS.
    n_parallel_jobs = int(trial.study.user_attrs.get('n_jobs', 1))
    is_parallel_optimization = n_parallel_jobs > 1

    # Use model-appropriate env count and derive timesteps from config
    n_envs = 8 if model_name == "ppo" else 2
    if is_parallel_optimization:
        n_envs = 1
    train_timesteps = calculate_total_timesteps(config, model_name, n_envs, timesteps_override)
    
    print(f"\n  Training for {train_timesteps:,} timesteps with {n_envs} envs...")
    
    # Create training env
    if is_parallel_optimization:
        print("  Parallel trials detected: using DummyVecEnv for stability")
        env = DummyVecEnv([make_env(config, i) for i in range(n_envs)])
    else:
        try:
            env = SubprocVecEnv([make_env(config, i) for i in range(n_envs)])
        except Exception as e:
            print(f"  ⚠ Failed to create SubprocVecEnv: {e}")
            print(f"  Falling back to DummyVecEnv...")
            env = DummyVecEnv([make_env(config, i) for i in range(n_envs)])
    
    # Create eval env
    eval_env = DummyVecEnv([make_env(config, 0)])
    
    model = create_model(model_name, config, env)
    
    # Train
    try:
        # Add callback for real-time progress
        training_callback = TrainingProgressCallback(
            trial_number=trial.number + 1,
            total_timesteps=train_timesteps,
            update_interval=10000  # Update every 10k steps
        )
        
        model.learn(
            total_timesteps=train_timesteps,
            callback=training_callback,
            progress_bar=False  # Use our custom progress
        )
    except Exception as e:
        print(f"  ⚠ Training failed: {e}")
        env.close()
        eval_env.close()
        return -1000.0  # Very bad score
    
    # ============================================================
    # EVALUATE MODEL
    # ============================================================
    
    print(f"  Evaluating...")
    
    n_eval_episodes = 20
    episode_rewards = []
    episode_lengths = []
    success_count = 0  # Count episodes where robot stands up
    
    for episode in range(n_eval_episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0
        steps = 0
        max_height = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward[0]
            steps += 1
            
            # Track max height reached
            if info[0].get('base_height', 0) > max_height:
                max_height = info[0].get('base_height', 0)
            
            if done[0]:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # Consider success if reached target height (0.31m)
        if max_height >= 0.31:
            success_count += 1
    
    # Cleanup
    env.close()
    eval_env.close()
    
    # ============================================================
    # COMPUTE METRICS
    # ============================================================
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    success_rate = success_count / n_eval_episodes
    
    trial_time = time.time() - trial_start_time
    
    print(f"\n  Results:")
    print(f"    Mean reward:   {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"    Success rate:  {success_rate:.1%} ({success_count}/{n_eval_episodes})")
    print(f"    Mean length:   {np.mean(episode_lengths):.0f} steps")
    print(f"    Trial time:    {trial_time/60:.1f} min")
    
    # Report intermediate values for pruning
    trial.report(mean_reward, step=0)
    
    # Check if trial should be pruned
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    # Return metric to MAXIMIZE
    # You can adjust this - e.g., weight success rate more:
    # return mean_reward * 0.7 + success_rate * 100 * 0.3
    return mean_reward


def run_optimization(n_trials=30, n_jobs=1, model_name="ppo", timesteps_override=None):
    """
    Run Optuna hyperparameter optimization
    
    Args:
        n_trials: Number of trials to run
        n_jobs: Number of parallel jobs (1 = sequential)
    """
    
    config = load_config()
    default_envs = 8 if model_name.lower() == "ppo" else 2
    timesteps_per_trial = calculate_total_timesteps(config, model_name, default_envs, timesteps_override)

    print("="*70)
    print("HYPERPARAMETER OPTIMIZATION - Go2 Self-Recovery")
    print("="*70)
    print(f"\nModel: {model_name.upper()}")
    if model_name.lower() == "sac":
        print("Optimizing: SAC hyperparameters")
    else:
        print("Optimizing: PPO hyperparameters")
    print(f"Trials: {n_trials}")
    print(f"Training per trial: {timesteps_per_trial:,} timesteps")
    if timesteps_override is not None:
        print("Formula: user-defined timestep budget")
    elif model_name.lower() == "ppo":
        block = config['ppo']['n_steps'] * default_envs
        print(f"Formula: nearest(2,500,400 / {block}) x {block} (rollout-aligned)")
    else:
        print("Formula: fixed target for HPO budget (~3.7M)")
    if n_jobs > 1:
        print(f"Running {n_jobs} trials in parallel")
        print(f"Total estimated timesteps: ~{(timesteps_per_trial * n_trials) / n_jobs:,.0f}")
    else:
        print(f"Total estimated timesteps: ~{timesteps_per_trial * n_trials:,}")
    print("\n" + "="*70 + "\n")
    
    # Create study
    study = optuna.create_study(
        study_name=f"go2_hyperparams_{model_name.lower()}",
        direction="maximize",  # Maximize reward
        sampler=TPESampler(seed=42),  # Tree-structured Parzen Estimator
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0)  # Prune bad trials early
    )
    
    # Store n_trials for progress tracking
    study.set_user_attr('n_trials', n_trials)
    study.set_user_attr('n_jobs', n_jobs)
    
    # Create progress callback (only for sequential runs)
    callbacks = []
    if n_jobs == 1:
        progress_callback = ProgressCallback(n_trials)
        callbacks = [progress_callback]
        print("📊 Real-time progress tracking enabled")
    else:
        print("⚠️  Progress summary only (parallel mode)")
    
    # Run optimization
    print("\n🚀 Starting optimization...\n")

    objective_fn = partial(objective, model_name=model_name, timesteps_override=timesteps_override)
    
    study.optimize(
        objective_fn,
        n_trials=n_trials,
        n_jobs=n_jobs,
        callbacks=callbacks,
        show_progress_bar=False
    )
    
    # Final summary for parallel runs
    if n_jobs > 1:
        print(f"\n{'='*70}")
        print(f"✅ All {n_trials} trials completed!")
        print(f"{'='*70}")
    
    # ============================================================
    # RESULTS
    # ============================================================
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    
    print("\n📊 Best Trial:")
    best_trial = study.best_trial
    print(f"  Trial number:  {best_trial.number}")
    print(f"  Mean reward:   {best_trial.value:.2f}")
    print(f"\n⚙️  Best Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key:15s}: {value}")
    
    # Save results
    output_dir = project_root / "logs" / "optuna" / model_name.lower() / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save study
    study_path = output_dir / "study.pkl"
    optuna.logging.get_logger("optuna").addHandler(optuna.logging.FileHandler(output_dir / "optuna.log"))
    
    # Save best params as YAML
    best_config_path = output_dir / "best_params.yml"
    with open(best_config_path, 'w') as f:
        yaml.dump(build_best_params_payload(model_name, best_trial, n_trials), f)

    history_csv_path = save_trial_history_csv(study, output_dir, model_name, timesteps_per_trial)
    plot_paths = save_optimization_plots(study, output_dir, model_name, timesteps_per_trial)
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"   - best_params.yml")
    print(f"   - {history_csv_path.name}")
    for plot_path in plot_paths:
        print(f"   - {plot_path.name}")
    
    # Print optimization history
    print("\n📈 Top 5 Trials:")
    trials_df = study.trials_dataframe()
    top_5 = trials_df.nlargest(5, 'value')[get_top_trials_columns(model_name)]
    print(top_5.to_string(index=False))
    
    print("\n⚡ Next Steps:")
    print("  1. Update config/train_config.yml with best parameters")
    print("  2. Run full training: mjpython src/scripts/train.py")
    print("  3. Evaluate: mjpython src/scripts/evaluate.py")
    
    return study


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize Go2 self-recovery hyperparameters")
    parser.add_argument('--trials', type=int, default=20, help='Number of optimization trials')
    parser.add_argument('--jobs', type=int, default=1, help='Number of parallel jobs (1=sequential)')
    parser.add_argument(
        '--timesteps',
        type=int,
        default=None,
        help='Override training timesteps per trial to compare smaller/larger HPO budgets'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='ppo',
        choices=['ppo', 'sac'],
        help='Model to optimize (ppo or sac)'
    )
    
    args = parser.parse_args()
    
    study = run_optimization(
        n_trials=args.trials,
        n_jobs=args.jobs,
        model_name=args.model,
        timesteps_override=args.timesteps,
    )
