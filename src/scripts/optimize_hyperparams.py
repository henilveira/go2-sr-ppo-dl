"""
Hyperparameter Optimization using Optuna
Optimizes: kp, kd, action_smoothing for Go2 self-recovery
"""

import sys
from pathlib import Path
import yaml
import numpy as np
from datetime import datetime, timedelta
import time
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch

from environment.go2_env import Go2Env


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


def objective(trial):
    """
    Optuna objective function - trains and evaluates model with suggested hyperparameters
    """
    
    trial_start_time = time.time()
    
    n_total_trials = trial.study.user_attrs.get('n_trials', '?')
    best_so_far = trial.study.best_value if len(trial.study.trials) > 1 and trial.study.best_value != float('-inf') else None
    
    print(f"\n{'='*70}")
    if best_so_far is not None:
        print(f"Trial {trial.number + 1}/{n_total_trials} | Best so far: {best_so_far:.2f}")
    else:
        print(f"Trial {trial.number + 1}/{n_total_trials}")
    print(f"{'='*70}")
    
    # Load base config
    config = load_config()
    
    # ============================================================
    # HYPERPARAMETERS TO OPTIMIZE
    # ============================================================
    
    # PD Controller gains
    kp = trial.suggest_float('kp', 25.0, 70.0, step=5.0)
    kd = trial.suggest_float('kd', 2.0, 8.0, step=0.5)
    
    # Action smoothing
    smoothing = trial.suggest_float('smoothing', 0.0, 0.4, step=0.05)
    
    # PPO learning rate (optional - comment out if you want to keep it fixed)
    # lr = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    
    # Update config with trial suggestions
    config['controller']['kp'] = kp
    config['controller']['kd'] = kd
    config['action']['smoothing'] = smoothing
    # config['ppo']['learning_rate'] = lr  # Uncomment if optimizing LR
    
    print(f"\nTesting hyperparameters:")
    print(f"  kp:        {kp:.1f}")
    print(f"  kd:        {kd:.1f}")
    print(f"  smoothing: {smoothing:.2f}")
    # print(f"  lr:        {lr:.2e}")
    
    # ============================================================
    # TRAIN MODEL (shorter training for HPO)
    # ============================================================
    
    # Use fewer envs and shorter training for faster optimization
    n_envs = 8  # Reduced from 12
    train_timesteps = 1_500_000  # ~10 min per trial on M1 Mac (reduced from 300k)
    
    print(f"\n  Training for {train_timesteps:,} timesteps with {n_envs} envs...")
    
    # Create training env
    try:
        env = SubprocVecEnv([make_env(config, i) for i in range(n_envs)])
    except Exception as e:
        print(f"  ⚠ Failed to create SubprocVecEnv: {e}")
        print(f"  Falling back to DummyVecEnv...")
        env = DummyVecEnv([make_env(config, i) for i in range(n_envs)])
    
    # Create eval env
    eval_env = DummyVecEnv([make_env(config, 0)])
    
    # Create PPO model
    ppo_config = config['ppo']
    model = PPO(
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
            'activation_fn': torch.nn.ReLU
        },
        verbose=0  # Suppress output
    )
    
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


def run_optimization(n_trials=30, n_jobs=1):
    """
    Run Optuna hyperparameter optimization
    
    Args:
        n_trials: Number of trials to run
        n_jobs: Number of parallel jobs (1 = sequential)
    """
    
    print("="*70)
    print("HYPERPARAMETER OPTIMIZATION - Go2 Self-Recovery")
    print("="*70)
    print(f"\nOptimizing: kp, kd, smoothing")
    print(f"Trials: {n_trials}")
    print(f"Training per trial: 200k timesteps (~10 min each)")
    if n_jobs > 1:
        print(f"Running {n_jobs} trials in parallel")
        print(f"Total estimated time: ~{n_trials * 10 / 60 / n_jobs:.1f} hours")
    else:
        print(f"Total estimated time: ~{n_trials * 10 / 60:.1f} hours")
    print("\n" + "="*70 + "\n")
    
    # Create study
    study = optuna.create_study(
        study_name="go2_hyperparams",
        direction="maximize",  # Maximize reward
        sampler=TPESampler(seed=42),  # Tree-structured Parzen Estimator
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0)  # Prune bad trials early
    )
    
    # Store n_trials for progress tracking
    study.set_user_attr('n_trials', n_trials)
    
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
    
    study.optimize(
        objective,
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
    output_dir = project_root / "logs" / "optuna" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save study
    study_path = output_dir / "study.pkl"
    optuna.logging.get_logger("optuna").addHandler(optuna.logging.FileHandler(output_dir / "optuna.log"))
    
    # Save best params as YAML
    best_config_path = output_dir / "best_params.yml"
    with open(best_config_path, 'w') as f:
        yaml.dump({
            'controller': {
                'kp': best_trial.params['kp'],
                'kd': best_trial.params['kd']
            },
            'action': {
                'smoothing': best_trial.params['smoothing']
            },
            'optimization_info': {
                'best_reward': best_trial.value,
                'trial_number': best_trial.number,
                'n_trials': n_trials
            }
        }, f)
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"   - best_params.yml")
    
    # Print optimization history
    print("\n📈 Top 5 Trials:")
    trials_df = study.trials_dataframe()
    top_5 = trials_df.nlargest(5, 'value')[['number', 'value', 'params_kp', 'params_kd', 'params_smoothing']]
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
    
    args = parser.parse_args()
    
    study = run_optimization(n_trials=args.trials, n_jobs=args.jobs)
