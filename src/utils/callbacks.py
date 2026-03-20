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


class StabilityMetricsCallback(BaseCallback):
    """
    Print stability metrics (support polygon, incircle, CoM margin) to terminal
    during training for quick inspection without TensorBoard.
    """
    
    def __init__(self, log_freq=10, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq  # Log every N rollouts
        self.rollout_count = 0
        self.stability_metrics = {
            'rewards': [],
            'polygon_areas': [],
            'incircle_radii': [],
            'safe_radii': [],
            'edge_margins': [],
            'streaks': [],
            'inside_safe': []
        }
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        """Collect and print stability metrics periodically"""
        self.rollout_count += 1
        
        # Collect metrics from current rollout
        if hasattr(self.locals, 'infos') and self.locals['infos']:
            for info in self.locals['infos']:
                if 'reward_breakdown' in info:
                    rb = info['reward_breakdown']
                    if 'R_stability' in rb:
                        self.stability_metrics['rewards'].append(float(rb['R_stability']))
                    if 'support_polygon_area' in rb:
                        self.stability_metrics['polygon_areas'].append(float(rb['support_polygon_area']))
                    if 'support_incircle_radius' in rb:
                        self.stability_metrics['incircle_radii'].append(float(rb['support_incircle_radius']))
                    if 'support_safe_radius' in rb:
                        self.stability_metrics['safe_radii'].append(float(rb['support_safe_radius']))
                    if 'support_edge_margin' in rb:
                        self.stability_metrics['edge_margins'].append(float(rb['support_edge_margin']))
                    if 'support_stability_streak' in rb:
                        self.stability_metrics['streaks'].append(float(rb['support_stability_streak']))
                    if 'support_com_inside_safe_circle' in rb:
                        self.stability_metrics['inside_safe'].append(float(rb['support_com_inside_safe_circle']))
        
        # Print periodically
        if self.rollout_count % self.log_freq == 0:
            self._print_stability_summary()
            # Clear metrics for next period
            for key in self.stability_metrics:
                self.stability_metrics[key] = []
    
    def _print_stability_summary(self):
        """Print a formatted stability metrics summary"""
        metrics = self.stability_metrics
        
        print(f"\n{'='*75}")
        print(f"[STABILITY METRICS] Rollout {self.rollout_count} | Steps: {self.num_timesteps:,}")
        print(f"{'='*75}")
        
        if metrics['rewards']:
            print(f"  R_stability:        {np.mean(metrics['rewards']):7.4f} ± {np.std(metrics['rewards']):.4f}")
        
        if metrics['inside_safe']:
            inside_rate = np.mean(metrics['inside_safe'])
            print(f"  CoM inside safe:    {inside_rate*100:6.1f}% of steps")
        
        if metrics['edge_margins']:
            edge_mean = np.mean(metrics['edge_margins'])
            edge_min = np.min(metrics['edge_margins'])
            print(f"  Edge margin:        mean={edge_mean:7.4f}m, min={edge_min:7.4f}m")
        
        if metrics['incircle_radii']:
            radius_mean = np.mean(metrics['incircle_radii'])
            radius_max = np.max(metrics['incircle_radii'])
            print(f"  Incircle radius:    mean={radius_mean:7.4f}m, max={radius_max:7.4f}m")
        
        if metrics['polygon_areas']:
            area_mean = np.mean(metrics['polygon_areas'])
            area_max = np.max(metrics['polygon_areas'])
            print(f"  Support area:       mean={area_mean:7.4f}m², max={area_max:7.4f}m²")
        
        if metrics['streaks']:
            streak_mean = np.mean(metrics['streaks'])
            streak_max = np.max(metrics['streaks'])
            print(f"  Stability streak:   mean={streak_mean:7.1f} steps, max={streak_max:7.1f} steps")
        
        print(f"{'='*75}\n")


class TerrainCurriculumCallback(BaseCallback):
    """Push global training progress into envs so terrain difficulty increases over time."""

    def __init__(self, total_timesteps, log_freq=10000, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = max(int(total_timesteps), 1)
        self.log_freq = log_freq
        self.latest_progress = 0.0
        self.terrain_counts = {}
        self.roughness_counts = {}
        self.incline_counts = {}

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_training_progress", 0.0)

    def _on_step(self) -> bool:
        self.latest_progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        self.training_env.env_method("set_training_progress", self.latest_progress)

        if hasattr(self.locals, 'infos') and self.locals['infos']:
            for info in self.locals['infos']:
                # Legacy terrain mode tracking
                terrain_mode = info.get('terrain_mode')
                if terrain_mode is not None:
                    self.terrain_counts[terrain_mode] = self.terrain_counts.get(terrain_mode, 0) + 1
                
                # New dynamic terrain mode tracking
                roughness = info.get('current_roughness')
                if roughness is not None:
                    self.roughness_counts[roughness] = self.roughness_counts.get(roughness, 0) + 1
                
                incline = info.get('current_pitch_deg')
                if incline is not None:
                    incline_key = f"{incline:.1f}°"
                    self.incline_counts[incline_key] = self.incline_counts.get(incline_key, 0) + 1

        if self.num_timesteps % self.log_freq == 0:
            self.logger.record("curriculum/training_progress", self.latest_progress)
            
            # Log terrain statistics
            total_terrain = sum(self.terrain_counts.values())
            if total_terrain > 0:
                for terrain_name, count in self.terrain_counts.items():
                    self.logger.record(f"terrain/{terrain_name}_ratio", count / total_terrain)
            
            # Log roughness statistics (new mode)
            total_roughness = sum(self.roughness_counts.values())
            if total_roughness > 0:
                mean_roughness = sum(k * v for k, v in self.roughness_counts.items()) / total_roughness
                self.logger.record("dynamic_terrain/mean_roughness", mean_roughness)
                self.logger.record("dynamic_terrain/max_roughness_seen", max(self.roughness_counts.keys()))
            
            # Log incline statistics (new mode)
            if len(self.incline_counts) > 0:
                print(f"\n  [Terrain Curriculum] Progress: {self.latest_progress*100:.1f}%")
                if total_roughness > 0:
                    print(f"    Mean Roughness: {mean_roughness:.1f} / 10")
                for incline_str, count in self.incline_counts.items():
                    ratio = count / total_roughness if total_roughness > 0 else 0
                    print(f"    Incline {incline_str}: {ratio*100:.1f}%")
                self.incline_counts = {}
            
            self.terrain_counts = {}
            self.roughness_counts = {}

        return True


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
                recovery_scores = []
                stagnation_steps = []
                spawn_retries = []
                spawn_valid_flags = []
                spawn_contacts = []
                spawn_penetrations = []
                support_rewards = []
                support_polygon_areas = []
                support_incircle_radii = []
                support_safe_radii = []
                support_edge_margins = []
                support_stability_streaks = []
                support_inside_safe_flags = []
                
                for info in self.locals['infos']:
                    if 'base_height' in info:
                        heights.append(info['base_height'])
                    if 'orientation_error' in info:
                        orientations.append(info['orientation_error'])
                    if 'recovery_score' in info:
                        recovery_scores.append(info['recovery_score'])
                    if 'steps_since_progress' in info:
                        stagnation_steps.append(info['steps_since_progress'])
                    if 'spawn_retry_count' in info:
                        spawn_retries.append(info['spawn_retry_count'])
                    if 'spawn_is_valid' in info:
                        spawn_valid_flags.append(info['spawn_is_valid'])
                    if 'spawn_contacts' in info:
                        spawn_contacts.append(info['spawn_contacts'])
                    if 'spawn_deepest_penetration' in info:
                        spawn_penetrations.append(info['spawn_deepest_penetration'])
                    if 'reward_breakdown' in info:
                        rb = info['reward_breakdown']
                        if 'curriculum_active' in rb:
                            success_flags.append(rb['curriculum_active'])
                        if 'R_stability' in rb:
                            support_rewards.append(rb['R_stability'])
                        if 'support_polygon_area' in rb:
                            support_polygon_areas.append(rb['support_polygon_area'])
                        if 'support_incircle_radius' in rb:
                            support_incircle_radii.append(rb['support_incircle_radius'])
                        if 'support_safe_radius' in rb:
                            support_safe_radii.append(rb['support_safe_radius'])
                        if 'support_edge_margin' in rb:
                            support_edge_margins.append(rb['support_edge_margin'])
                        if 'support_stability_streak' in rb:
                            support_stability_streaks.append(rb['support_stability_streak'])
                        if 'support_com_inside_safe_circle' in rb:
                            support_inside_safe_flags.append(rb['support_com_inside_safe_circle'])
                
                # Log environment-specific metrics
                if len(heights) > 0:
                    self.logger.record("env/mean_height", np.mean(heights))
                    self.logger.record("env/max_height", np.max(heights))
                
                if len(orientations) > 0:
                    self.logger.record("env/mean_orientation_error", np.mean(orientations))

                if len(recovery_scores) > 0:
                    self.logger.record("env/mean_recovery_score", np.mean(recovery_scores))

                if len(stagnation_steps) > 0:
                    self.logger.record("env/mean_stagnation_steps", np.mean(stagnation_steps))
                
                if len(success_flags) > 0:
                    self.logger.record("env/curriculum_activation_rate", np.mean(success_flags))

                if len(spawn_retries) > 0:
                    self.logger.record("spawn/mean_retries", np.mean(spawn_retries))
                    self.logger.record("spawn/max_retries", np.max(spawn_retries))

                if len(spawn_valid_flags) > 0:
                    self.logger.record("spawn/valid_rate", np.mean(spawn_valid_flags))

                if len(spawn_contacts) > 0:
                    self.logger.record("spawn/mean_contacts", np.mean(spawn_contacts))

                if len(spawn_penetrations) > 0:
                    self.logger.record("spawn/min_contact_dist", np.min(spawn_penetrations))

                # Support polygon / stability metrics
                if len(support_rewards) > 0:
                    self.logger.record("stability/reward", np.mean(support_rewards))

                if len(support_polygon_areas) > 0:
                    self.logger.record("stability/polygon_area_mean", np.mean(support_polygon_areas))

                if len(support_incircle_radii) > 0:
                    self.logger.record("stability/incircle_radius_mean", np.mean(support_incircle_radii))

                if len(support_safe_radii) > 0:
                    self.logger.record("stability/safe_radius_mean", np.mean(support_safe_radii))

                if len(support_edge_margins) > 0:
                    self.logger.record("stability/edge_margin_mean", np.mean(support_edge_margins))
                    self.logger.record("stability/edge_margin_min", np.min(support_edge_margins))

                if len(support_stability_streaks) > 0:
                    self.logger.record("stability/streak_mean", np.mean(support_stability_streaks))
                    self.logger.record("stability/streak_max", np.max(support_stability_streaks))

                if len(support_inside_safe_flags) > 0:
                    self.logger.record("stability/inside_safe_rate", np.mean(support_inside_safe_flags))
            
            # Update last log time
            self.last_log_time = current_time
            self.last_log_step = self.num_timesteps
        
        return True
