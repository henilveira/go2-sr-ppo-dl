"""
Custom callbacks for training monitoring
"""

import numpy as np
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
