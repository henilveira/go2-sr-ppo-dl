"""
Custom callbacks for training monitoring
"""

import numpy as np
import time
from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggerCallback(BaseCallback):
    """
    Custom callback to log reward statistics to terminal during training
    """
    
    def __init__(self, log_freq=10, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq  # Log every N rollouts
        self.episode_rewards = []
        self.episode_lengths = []
        self.rollout_count = 0
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout
        Log reward statistics to terminal
        """
        self.rollout_count += 1
        
        # Get episode rewards from logger
        if len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            ep_lengths = [ep_info["l"] for ep_info in self.model.ep_info_buffer]
            
            if len(ep_rewards) > 0 and self.rollout_count % self.log_freq == 0:
                mean_reward = np.mean(ep_rewards)
                std_reward = np.std(ep_rewards)
                min_reward = np.min(ep_rewards)
                max_reward = np.max(ep_rewards)
                mean_length = np.mean(ep_lengths)
                
                print(f"\n{'='*70}")
                print(f"Rollout {self.rollout_count} | Steps: {self.num_timesteps:,}")
                print(f"{'='*70}")
                print(f"  Episode Reward:  {mean_reward:8.2f} ± {std_reward:.2f}")
                print(f"  Min/Max Reward:  {min_reward:8.2f} / {max_reward:.2f}")
                print(f"  Episode Length:  {mean_length:8.1f}")
                print(f"{'='*70}\n")


class CurriculumMonitorCallback(BaseCallback):
    """
    Monitor curriculum learning activation during training
    """
    
    def __init__(self, log_freq=100, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.curriculum_activations = []
        
    def _on_step(self) -> bool:
        # Check if we can access info from environment
        if hasattr(self.locals, 'infos') and self.locals['infos']:
            for info in self.locals['infos']:
                if 'reward_breakdown' in info:
                    is_active = info['reward_breakdown'].get('curriculum_active', 0)
                    self.curriculum_activations.append(is_active)
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Log curriculum statistics"""
        if len(self.curriculum_activations) > 0:
            activation_rate = np.mean(self.curriculum_activations)
            
            if self.num_timesteps % (self.log_freq * 1024) < 1024:  # Log periodically
                print(f"  [Curriculum] Activation rate: {activation_rate*100:.1f}%")
            
            # Clear for next period
            self.curriculum_activations = []


class TensorBoardMetricsCallback(BaseCallback):
    """
    Log custom metrics to TensorBoard during training
    - Training time (wall clock)
    - Steps per second (SPS)
    - Custom environment metrics
    """
    
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq  # Log every N steps
        self.start_time = None
        self.last_log_time = None
        self.last_log_step = 0
        
    def _on_training_start(self) -> None:
        """Called before the first step"""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_log_step = 0
    
    def _on_step(self) -> bool:
        """Called at every step"""
        
        # Log periodically
        if self.num_timesteps % self.log_freq == 0:
            current_time = time.time()
            
            # ==================== TIME METRICS ====================
            # Total training time (hours)
            total_time = (current_time - self.start_time) / 3600
            self.logger.record("time/total_hours", total_time)
            
            # Time since last log (minutes)
            time_delta = (current_time - self.last_log_time) / 60
            self.logger.record("time/delta_minutes", time_delta)
            
            # Steps per second (recent)
            steps_delta = self.num_timesteps - self.last_log_step
            sps = steps_delta / (current_time - self.last_log_time) if current_time > self.last_log_time else 0
            self.logger.record("time/steps_per_second", sps)
            
            # ==================== EPISODE METRICS ====================
            if len(self.model.ep_info_buffer) > 0:
                ep_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                ep_lengths = [ep_info["l"] for ep_info in self.model.ep_info_buffer]
                
                # Reward statistics
                self.logger.record("rollout/ep_rew_std", np.std(ep_rewards))
                self.logger.record("rollout/ep_rew_min", np.min(ep_rewards))
                self.logger.record("rollout/ep_rew_max", np.max(ep_rewards))
                
                # Episode length statistics
                self.logger.record("rollout/ep_len_std", np.std(ep_lengths))
                
            # ==================== ENVIRONMENT METRICS ====================
            # Extract custom metrics from info dict (if available)
            if hasattr(self.locals, 'infos') and self.locals['infos']:
                heights = []
                orientations = []
                success_flags = []
                
                for info in self.locals['infos']:
                    if 'base_height' in info:
                        heights.append(info['base_height'])
                    if 'orientation_error' in info:
                        orientations.append(info['orientation_error'])
                    if 'reward_breakdown' in info:
                        rb = info['reward_breakdown']
                        if 'curriculum_active' in rb:
                            success_flags.append(rb['curriculum_active'])
                
                # Log environment-specific metrics
                if len(heights) > 0:
                    self.logger.record("env/mean_height", np.mean(heights))
                    self.logger.record("env/max_height", np.max(heights))
                
                if len(orientations) > 0:
                    self.logger.record("env/mean_orientation_error", np.mean(orientations))
                
                if len(success_flags) > 0:
                    self.logger.record("env/curriculum_activation_rate", np.mean(success_flags))
            
            # Update last log time
            self.last_log_time = current_time
            self.last_log_step = self.num_timesteps
        
        return True
