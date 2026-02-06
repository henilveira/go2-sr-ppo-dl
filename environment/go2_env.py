"""
Quadruped Self-Recovery Environment - Unitree Go2 + MuJoCo
Based on paper: "Self-Recovery of Quadrupedal Robot Using DRL" (2024)
Adapted for Go2 robot using MuJoCo physics engine
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from pathlib import Path


class Go2Env(gym.Env):
    """
    Ambiente Gym para self-recovery do Unitree Go2 usando MuJoCo
    
    Observation space: 30 dimensions
    - 12: joint positions
    - 12: joint velocities  
    - 3: base orientation (R^-1 · g)
    - 3: base angular velocity
    
    Action space: 12 dimensions
    - Target joint positions (normalized to [-1, 1])
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}
    
    def __init__(self, config, render_mode=None):
        super().__init__()
        
        self.config = config
        self.render_mode = render_mode
        
        # Timesteps - must be defined BEFORE _load_model()
        self.dyn_dt = self.config.get('simulation', {}).get('dyn_dt', 0.001)  # Physics timestep
        self.con_dt = self.config.get('simulation', {}).get('con_dt', 0.01)   # Control timestep
        self.n_substeps = int(self.con_dt / self.dyn_dt)  # How many physics steps per control step
        
        # Load MuJoCo model
        self._load_model()
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(30,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )
        
        # Initialize state
        self.prev_action = np.zeros(12)
        self.current_action = np.zeros(12)  # For smoothing
        self.step_count = 0
        
        # Action smoothing factor (0 = no smoothing, 1 = full smoothing)
        # Higher value = smoother but slower response
        self.action_smoothing = self.config.get('action', {}).get('smoothing', 0.35)
        
        # For rendering
        self.viewer = None
        if self.render_mode == 'human':
            # Will be initialized on first render() call
            pass
            
    def _load_model(self):
        """Load Go2 MuJoCo model"""
        # Path to Go2 XML model
        model_path = self.config['robot'].get('model_path', 'assets/mujoco/unitree_go2/scene.xml')
        
        # Make path absolute if relative
        if not Path(model_path).is_absolute():
            # Resolve relative to config directory
            config_dir = Path(__file__).parent.parent / 'config'
            model_path = (config_dir / model_path).resolve()
        
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Go2 model not found at {model_path}\n"
                "Download from: https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2"
            )
        
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = self.dyn_dt

        self.data = mujoco.MjData(self.model)
        
        # Get joint IDs (Go2 has 12 actuated joints)
        self._setup_joints()
        
    def _setup_joints(self):
        """Map controllable joints"""
        # Go2 joint names (em ordem):
        # FR: hip, thigh, calf (Front Right)
        # FL: hip, thigh, calf (Front Left)  
        # RR: hip, thigh, calf (Rear Right)
        # RL: hip, thigh, calf (Rear Left)
        
        joint_names = [
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
        ]
        
        self.joint_ids = []
        self.joint_limits = []
        
        for name in joint_names:
            try:
                joint_id = self.model.joint(name).id
                self.joint_ids.append(joint_id)
                
                # Get joint limits from model
                jnt_range = self.model.jnt_range[joint_id]
                self.joint_limits.append((jnt_range[0], jnt_range[1]))
                
            except KeyError:
                print(f"Warning: Joint {name} not found in model")
        
        print(f"Found {len(self.joint_ids)} controllable joints")
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state - robot on its back with legs tucked"""
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # ============================================================
        # INITIAL POSITION: Robot on its back (belly up), legs tucked
        # This is the self-recovery starting position from the paper
        # ============================================================
        
        # Base position - LOW on the ground
        initial_height = 0.12  # Slightly higher to account for legs
        self.data.qpos[0:3] = [0, 0, initial_height]
        
        # Orientation: ALWAYS on back (upside down) - roll = π
        # Small random variations only
        roll = np.pi + np.random.uniform(-0.1, 0.1)  # ~180° (on back)
        pitch = np.random.uniform(-0.1, 0.1)  # Small pitch variation
        yaw = np.random.uniform(-np.pi, np.pi)  # Any yaw is fine
        quat = self._euler_to_quat([roll, pitch, yaw])
        self.data.qpos[3:7] = quat
        
        # Joint positions: Legs TUCKED IN (bent towards body)
        # When on back, legs should be folded up
        # Thigh ~2.0 (bent up), Calf ~-2.0 (bent back towards thigh)
        tucked_config = np.array([
            0.0, 1.8, -2.4,   # FR: hip neutral, thigh up, calf tucked
            0.0, 1.8, -2.4,   # FL
            0.0, 1.8, -2.4,   # RR  
            0.0, 1.8, -2.4    # RL
        ])
        
        # Add small random noise to joint positions
        if self.config['training'].get('random_joint_positions', True):
            noise = np.random.uniform(-0.2, 0.2, size=12)
            joint_pos = tucked_config + noise
            
            # Clip to joint limits
            for i, (lower, upper) in enumerate(self.joint_limits):
                joint_pos[i] = np.clip(joint_pos[i], lower, upper)
            
            self.data.qpos[7:19] = joint_pos
        else:
            self.data.qpos[7:19] = tucked_config
        
        # Zero velocities
        self.data.qvel[:] = 0
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # CRITICAL: Let robot settle on ground (like original - 100 steps)
        # This prevents weird floating/unstable initial states
        for _ in range(100):
            # Apply small torques to help settle
            self.data.ctrl[:] = 0
            mujoco.mj_step(self.model, self.data)
        
        # Reset tracking variables
        self.prev_action = np.zeros(12)
        self.current_action = np.zeros(12)  # Reset smoothed action too
        self.step_count = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
        
    def step(self, action):
        """Execute one step in the environment"""
        # Apply action smoothing (exponential moving average)
        # This reduces jerky movements by gradually transitioning between actions
        smoothed_action = (1 - self.action_smoothing) * action + self.action_smoothing * self.current_action
        self.current_action = smoothed_action
        
        # Scale action from [-1, 1] to actual joint limits
        scaled_action = self._scale_action(smoothed_action)
        
        for _ in range(self.n_substeps):
            self._apply_pd_control(scaled_action)
            mujoco.mj_step(self.model, self.data)
        
        # Get observation
        observation = self._get_observation()
        
        # Compute reward (use original action for reward, not smoothed)
        reward, reward_info = self._compute_reward(observation, action)
        
        # Check termination
        terminated = self._is_terminated(observation)
        truncated = self.step_count >= self.config['training']['max_episode_steps']
        
        # Update tracking
        self.prev_action = action
        self.step_count += 1
        
        info = self._get_info()
        info['reward_breakdown'] = reward_info
        
        # Render if needed
        if self.render_mode == 'human':
            self.render()
        
        return observation, reward, terminated, truncated, info
        
    def _get_observation(self):
        """
        Get 30-dimensional observation
        Paper Section II.B, Table I
        """
        # Joint positions (12) - indices 0-11
        joint_positions = self.data.qpos[7:19].copy()  # Skip base pose (7 values)
        
        # Joint velocities (12) - indices 12-23
        joint_velocities = self.data.qvel[6:18].copy()  # Skip base velocity (6 values)
        
        # Base orientation (3) - indices 24-26
        # Paper eq. (6): θ_B = R^-1 · g
        base_quat = self.data.qpos[3:7]
        rot_matrix = self._quat_to_matrix(base_quat)
        gravity_vec = np.array([0, 0, -1])
        base_orientation = rot_matrix.T @ gravity_vec
        
        # Base angular velocity (3) - indices 27-29
        base_angular_vel = self.data.qvel[3:6].copy()  # Angular velocity in world frame
        
        # Concatenate
        obs = np.concatenate([
            joint_positions,      # 12
            joint_velocities,     # 12
            base_orientation,     # 3
            base_angular_vel      # 3
        ])  # Total: 30
        
        # Add noise (paper Section II.B)
        obs = self._add_observation_noise(obs)
        
        # Normalize to [-1, 1] (paper eq. 7)
        obs = self._normalize_observation(obs)
        
        return obs.astype(np.float32)
        
    def _add_observation_noise(self, obs):
        """
        Add sensor noise to observation
        Paper: "to replicate real robot we introduce noisy observation"
        - Joint positions: ±0.1 rad
        - Joint velocities: ±1.0 rad/s
        - Base angular velocity: ±0.2 rad/s
        """
        if not self.config['observation'].get('add_noise', True):
            return obs
            
        noise_config = self.config['observation']['noise']
        
        # Joint positions noise
        obs[0:12] += np.random.uniform(
            -noise_config['joint_positions'],
            noise_config['joint_positions'],
            size=12
        )
        
        # Joint velocities noise
        obs[12:24] += np.random.uniform(
            -noise_config['joint_velocities'],
            noise_config['joint_velocities'],
            size=12
        )
        
        # Base angular velocity noise (indices 27-29)
        obs[27:30] += np.random.uniform(
            -noise_config['base_angular_velocity'],
            noise_config['base_angular_velocity'],
            size=3
        )
        
        return obs
        
    def _normalize_observation(self, obs):
        """
        Normalize observation to [-1, 1]
        Paper eq. (7): f(x) = y_min + (y_max - y_min)/(x_max - x_min) * (x - x_min)
        """
        norm_config = self.config['observation']['normalization']
        
        # Joint positions (0-11)
        obs[0:12] = self._normalize_values(
            obs[0:12],
            norm_config['joint_pos_min'],
            norm_config['joint_pos_max'],
            -1.0, 1.0
        )
        
        # Joint velocities (12-23)
        obs[12:24] = self._normalize_values(
            obs[12:24],
            norm_config['joint_vel_min'],
            norm_config['joint_vel_max'],
            -1.0, 1.0
        )
        
        # Base orientation (24-26) - already in [-1, 1] range
        obs[24:27] = np.clip(obs[24:27], -1.0, 1.0)
        
        # Base angular velocity (27-29)
        obs[27:30] = self._normalize_values(
            obs[27:30],
            norm_config['base_ang_vel_min'],
            norm_config['base_ang_vel_max'],
            -1.0, 1.0
        )
        
        return obs
        
    def _normalize_values(self, values, x_min, x_max, y_min, y_max):
        """Apply eq. (7) from paper"""
        return y_min + (y_max - y_min) / (x_max - x_min) * (values - x_min)
        
    def _scale_action(self, action):
        """
        Scale action from [-1, 1] to actual joint positions.
        
        For self-righting, the robot needs FULL range of motion to flip over.
        Using full joint limits but with smoothing applied in step().
        """
        scaled = np.zeros(12)
        action_scale = self.config.get('action', {}).get('scale', 1.0)

        for i in range(12):
            lower, upper = self.joint_limits[i]
            mid = (lower + upper) * 0.5
            half_range = (upper - lower) * 0.5 * action_scale
            # Map [-1, 1] to [mid - half_range, mid + half_range]
            scaled[i] = mid + action[i] * half_range

        return scaled
        
    def _apply_pd_control(self, target_positions):
        """
        Apply PD controller to reach target joint positions
        Paper Fig. 1: Policy network outputs target positions to PD controller
        """
        # PD gains from config
        kp = self.config.get('controller', {}).get('kp', 30.0)
        kd = self.config.get('controller', {}).get('kd', 5.0)
        max_torque = self.config.get('controller', {}).get('max_torque', 23.5)
        
        for i, joint_id in enumerate(self.joint_ids):
            # Current state
            current_pos = self.data.qpos[7 + i]
            current_vel = self.data.qvel[6 + i]
            
            # PD control
            error = target_positions[i] - current_pos
            torque = kp * error - kd * current_vel
            
            # Clip torque to actuator limits
            torque = np.clip(torque, -max_torque, max_torque)
            
            # Apply torque
            self.data.ctrl[i] = torque
            
    def _compute_reward(self, obs, action):
        """
        Compute reward using paper's reward function
        Import from reward_manager for modularity
        """
        import sys
        from pathlib import Path
        
        # Add src to path if not already there
        src_path = str(Path(__file__).parent.parent / 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from rewards.reward_manager import RewardManager
        
        if not hasattr(self, 'reward_manager'):
            self.reward_manager = RewardManager(self.config, self.model, self.data)
        
        return self.reward_manager.compute(
            obs, action, self.prev_action, self._get_info()
        )
        
    def _is_terminated(self, obs):
        """
        Check if episode should terminate
        Paper: "The only termination condition was the maximum number of steps"
        So we return False here, truncation handles max steps
        """
        return False
        
    def _get_info(self):
        """Get additional info for reward computation"""
        # Base height
        base_height = self.data.qpos[2]
        
        # Foot contacts (for foot contact reward)
        # Need to check which geoms are in contact with ground
        feet_contacts = self._get_feet_contacts()
        
        # Base velocity
        base_linear_vel = self.data.qvel[0:3]
        
        return {
            'base_height': base_height,
            'feet_contacts': feet_contacts,
            'base_linear_velocity': base_linear_vel,
            'step': self.step_count
        }
        
    def _get_feet_contacts(self):
        """
        Check which feet are in contact with ground
        Returns list of 4 booleans [FR, FL, RR, RL]
        """
        # Feet geom names in Go2 model (from mujoco_menagerie)
        feet_geom_names = [
            'FR',  # Front Right foot
            'FL',  # Front Left foot
            'RR',  # Rear Right foot
            'RL'   # Rear Left foot
        ]
        
        contacts = [False] * 4
        
        # Get floor geom ID (usually 0, but let's be safe)
        try:
            floor_id = self.model.geom('floor').id
        except KeyError:
            floor_id = 0  # Assume floor is geom 0
        
        # Check all contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Get geom IDs in contact
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            # Check if it's a foot-ground contact
            for j, foot_name in enumerate(feet_geom_names):
                try:
                    foot_geom_id = self.model.geom(foot_name).id
                    
                    # Check if foot is in contact with ground
                    if (geom1 == foot_geom_id and geom2 == floor_id) or \
                       (geom2 == foot_geom_id and geom1 == floor_id):
                        contacts[j] = True
                except KeyError:
                    pass  # Geom name not found
        
        return contacts
        
    def _quat_to_matrix(self, quat):
        """Convert quaternion to rotation matrix"""
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        return mat.reshape(3, 3)
        
    def _euler_to_quat(self, euler):
        """Convert Euler angles (roll, pitch, yaw) to quaternion"""
        quat = np.zeros(4)
        # MuJoCo uses wxyz format
        cy = np.cos(euler[2] * 0.5)
        sy = np.sin(euler[2] * 0.5)
        cp = np.cos(euler[1] * 0.5)
        sp = np.sin(euler[1] * 0.5)
        cr = np.cos(euler[0] * 0.5)
        sr = np.sin(euler[0] * 0.5)
        
        quat[0] = cr * cp * cy + sr * sp * sy  # w
        quat[1] = sr * cp * cy - cr * sp * sy  # x
        quat[2] = cr * sp * cy + sr * cp * sy  # y
        quat[3] = cr * cp * sy - sr * sp * cy  # z
        
        return quat
        
    def close(self):
        """Clean up resources"""
        if self.viewer is not None:
            try:
                self.viewer.close()
            except:
                pass
            
    def render(self):
        """Render the environment"""
        if self.render_mode == 'rgb_array':
            # Return RGB array for video recording
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            return renderer.render()
        
        elif self.render_mode == 'human':
            # Use passive viewer for interactive visualization
            if self.viewer is None:
                # Launch viewer on first call
                try:
                    # Try new API first (MuJoCo 3.0+)
                    from mujoco import viewer
                    self.viewer = viewer.launch_passive(self.model, self.data)
                except (AttributeError, ImportError):
                    # Fallback: use handle_passive for older versions
                    try:
                        self.viewer = mujoco.viewer.launch_passive(
                            model=self.model, 
                            data=self.data
                        )
                    except:
                        # Last resort: print warning and disable rendering
                        print("Warning: Could not launch MuJoCo viewer. Rendering disabled.")
                        self.render_mode = None
                        return None
            
            # Sync viewer with simulation data
            if self.viewer is not None:
                try:
                    self.viewer.sync()
                except:
                    pass


# Wrapper for vectorized environments (for parallel training)
def make_go2_env(config, rank=0):
    """
    Create a single Go2 environment instance
    Used for creating multiple parallel environments
    """
    def _init():
        env = Go2RecoveryEnv(config)
        return env
    return _init