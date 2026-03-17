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

from environment.terrain import TerrainCurriculum


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

        self.training_progress = 0.0
        self.current_terrain = 'default'
        self.terrain_curriculum_config = self.config.get('terrain_curriculum', {})
        
        # Initialize terrain curriculum for dynamic generation
        self.terrain_curriculum = TerrainCurriculum(num_roughness_levels=10)
        self.current_roughness = 0
        self.current_pitch_deg = 0.0
        
        # Cache for generated height fields to avoid regenerating each episode
        self._hfield_cache = {}
        self._cache_seed_offset = np.random.randint(0, 10000)
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
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
        self.best_recovery_score = 0.0
        self.steps_since_progress = 0
        self.spawn_retry_count = 0
        self.spawn_is_valid = True
        self.spawn_ncon = 0
        self.spawn_deepest_penetration = 0.0
        
        # Action smoothing factor (0 = no smoothing, 1 = full smoothing)
        # Higher value = smoother but slower response
        self.action_smoothing = self.config.get('action', {}).get('smoothing', 0.35)
        
        # For rendering
        self.viewer = None
        if self.render_mode == 'human':
            # Will be initialized on first render() call
            pass

    def set_training_progress(self, progress):
        """Update global training progress for terrain curriculum callbacks."""
        self.training_progress = float(np.clip(progress, 0.0, 1.0))
            
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
        """
        Map joints and actuators correctly.
        IMPORTANTE: A ordem dos joints (qpos) pode ser diferente da ordem dos actuators (ctrl)!
        """
        # Definir ordem das pernas e links
        self.legs = ['FR', 'FL', 'RR', 'RL']
        self.links = ['hip', 'thigh', 'calf']
        
        # Construir nomes na ordem que queremos usar na policy
        self.joint_names = []
        self.actuator_names = []
        
        for leg in self.legs:
            for link in self.links:
                self.joint_names.append(f"{leg}_{link}_joint")
                self.actuator_names.append(f"{leg}_{link}")
        
        # Mapear joints -> qpos addresses
        self.joint_ids = []
        self.joint_qpos_addr = []
        self.joint_qvel_addr = []
        self.joint_limits = []
        
        for name in self.joint_names:
            try:
                joint_id = self.model.joint(name).id
                self.joint_ids.append(joint_id)
                self.joint_qpos_addr.append(self.model.jnt_qposadr[joint_id])
                self.joint_qvel_addr.append(self.model.jnt_dofadr[joint_id])
                
                # Get joint limits from model
                jnt_range = self.model.jnt_range[joint_id]
                self.joint_limits.append((jnt_range[0], jnt_range[1]))
                
            except KeyError:
                print(f"Warning: Joint {name} not found in model")
        
        # Mapear actuators -> ctrl indices (PODE SER ORDEM DIFERENTE!)
        self.actuator_ids = []
        for name in self.actuator_names:
            try:
                actuator_id = self.model.actuator(name).id
                self.actuator_ids.append(actuator_id)
            except KeyError:
                print(f"Warning: Actuator {name} not found in model")
        
        # Debug: mostra o mapeamento
        print(f"\n{'='*50}")
        print("JOINT/ACTUATOR MAPPING:")
        print(f"{'='*50}")
        for i, (jname, aname) in enumerate(zip(self.joint_names, self.actuator_names)):
            qpos = self.joint_qpos_addr[i]
            ctrl = self.actuator_ids[i]
            print(f"  [{i:2d}] {jname:20s} -> qpos[{qpos}], ctrl[{ctrl}]")
        print(f"{'='*50}\n")
        
        print(f"Found {len(self.joint_ids)} joints, {len(self.actuator_ids)} actuators")

    def _sample_terrain_mode(self):
        """Sample terrain from config-driven curriculum (dynamic or legacy staged)."""
        curriculum = self.terrain_curriculum_config
        if not curriculum.get('enabled', False):
            terrain_name = curriculum.get('default_terrain', 'default')
            self.current_terrain = terrain_name
            self.current_roughness = 0
            self.current_pitch_deg = 0.0
            return terrain_name

        mode = curriculum.get('mode', 'stages')
        if mode == 'dynamic_roughness':
            terrain_config = self.terrain_curriculum.get_terrain_by_progress(self.training_progress)
            self.current_terrain = terrain_config['name']
            self.current_roughness = terrain_config['roughness_level']
            self.current_pitch_deg = terrain_config['pitch_deg']
            return terrain_config

        stages = curriculum.get('stages', [])
        if not stages:
            terrain_name = curriculum.get('default_terrain', 'default')
            self.current_terrain = terrain_name
            self.current_roughness = 0
            self.current_pitch_deg = 0.0
            return terrain_name

        active_stage = stages[-1]
        for stage in stages:
            if self.training_progress <= stage.get('until_progress', 1.0):
                active_stage = stage
                break

        weights = active_stage.get('weights', {'default': 1.0})
        terrain_names = list(weights.keys())
        probabilities = np.asarray(list(weights.values()), dtype=np.float64)
        probabilities = probabilities / probabilities.sum()
        terrain_name = str(np.random.choice(terrain_names, p=probabilities))

        pitch_deg = 0.0
        if '_l' in terrain_name:
            try:
                level = int(terrain_name.rsplit('_l', 1)[1])
                if terrain_name.startswith('flat_inclined_l') or terrain_name.startswith('rocky_inclined_l') or terrain_name.startswith('rough_l'):
                    pitch_deg = -(level + 1.0)
            except ValueError:
                pitch_deg = 0.0
        elif 'inclined' in terrain_name or terrain_name == 'rough':
            pitch_deg = -10.0

        self.current_terrain = terrain_name
        self.current_roughness = 0
        self.current_pitch_deg = pitch_deg
        return terrain_name

    def _place_robot_for_terrain(self, terrain_mode):
        """Place the robot over the selected terrain patch."""
        spawn_lift = 0.0
        # Handle both old string-based and new dict-based terrain modes
        if isinstance(terrain_mode, str):
            incline_center = lambda deg: -0.05 + 1.5 * np.sin(np.deg2rad(deg))

            # Level-tagged staged terrains (e.g., rough_l3, rocky_inclined_l7)
            if terrain_mode.startswith('flat_inclined_l'):
                level = int(np.clip(int(terrain_mode.rsplit('_l', 1)[1]), 0, 8))
                incline_deg = float(level + 1)
                spawn = {
                    'pos': [3.25 * level, 12.0, 0.24 + incline_center(incline_deg) + spawn_lift],
                    'pitch_deg': -incline_deg,
                    'settle_steps': 80,
                }
            elif terrain_mode.startswith('rough_l'):
                level = int(np.clip(int(terrain_mode.rsplit('_l', 1)[1]), 0, 8))
                incline_deg = float(level + 1)
                spawn = {
                    'pos': [3.0 * level, 9.0, 0.24 + incline_center(incline_deg) + spawn_lift],
                    'pitch_deg': -incline_deg,
                    'settle_steps': 80,
                }
            elif terrain_mode.startswith('rocky_l'):
                level = int(np.clip(int(terrain_mode.rsplit('_l', 1)[1]), 0, 9))
                spawn = {
                    'pos': [3.25 * level, 3.0, 0.40 + spawn_lift],
                    'pitch_deg': 0.0,
                    'settle_steps': 80,
                }
            elif terrain_mode.startswith('rocky_inclined_l'):
                level = int(np.clip(int(terrain_mode.rsplit('_l', 1)[1]), 0, 9))
                incline_deg = float(level + 1)
                spawn = {
                    'pos': [3.25 * level, 6.0, 0.24 + incline_center(incline_deg) + spawn_lift],
                    'pitch_deg': -incline_deg,
                    'settle_steps': 80,
                }
            else:
            # Legacy mode - keep old behavior for compatibility
                spawn_map = {
                    'flat': {'pos': [0.0, 0.0, 0.12 + spawn_lift], 'pitch_deg': 0.0, 'settle_steps': 100},
                    'flat_inclined': {'pos': [3.25, 12.0, 0.30 + spawn_lift], 'pitch_deg': -10.0, 'settle_steps': 80},
                    'rocky': {'pos': [0.0, 3.0, 0.40 + spawn_lift], 'pitch_deg': 0.0, 'settle_steps': 80},
                    'rocky_inclined': {'pos': [3.25, 6.0, 0.30 + spawn_lift], 'pitch_deg': -10.0, 'settle_steps': 80},
                    'rough': {'pos': [3.25, 9.0, 0.30 + spawn_lift], 'pitch_deg': -10.0, 'settle_steps': 80},
                    # Backward-compatible aliases
                    'default': {'pos': [0.0, 0.0, 0.12 + spawn_lift], 'pitch_deg': 0.0, 'settle_steps': 100},
                    'inclined': {'pos': [3.25, 6.0, 0.30 + spawn_lift], 'pitch_deg': -10.0, 'settle_steps': 80},
                    'smooth_inclined': {'pos': [3.25, 12.0, 0.30 + spawn_lift], 'pitch_deg': -10.0, 'settle_steps': 80},
                }
                spawn = spawn_map.get(terrain_mode, spawn_map['flat'])
        else:
            # New mode - use dynamic terrain config dict
            roughness = terrain_mode.get('roughness_level', 0)
            pitch_deg = terrain_mode.get('pitch_deg', 0.0)
            jitter_xy = float(self.config.get('training', {}).get('spawn_xy_jitter', 0.35))
            x_offset = np.random.uniform(-jitter_xy, jitter_xy)
            y_offset = np.random.uniform(-jitter_xy, jitter_xy)
            
            # Generate or retrieve height field
            self._update_terrain_heightfield(roughness)
            
            # Calculate spawn position based on pitch angle
            # Adjust height to compensate for incline
            pitch_rad = np.deg2rad(pitch_deg)
            base_height = 0.12 + spawn_lift
            z_adjust = 3.0 * abs(np.sin(pitch_rad))
            spawn = {
                'pos': [x_offset, y_offset, base_height + z_adjust],
                'pitch_deg': pitch_deg,
                'settle_steps': 100,
            }

        self.data.qpos[0:3] = spawn['pos']

        roll = np.pi + np.random.uniform(-0.1, 0.1)
        pitch = np.deg2rad(spawn['pitch_deg']) + np.random.uniform(-0.1, 0.1)
        yaw = np.random.uniform(-np.pi, np.pi)
        quat = self._euler_to_quat([roll, pitch, yaw])
        self.data.qpos[3:7] = quat

        tucked_config = np.array([
            0.0, 1.8, -2.4,
            0.0, 1.8, -2.4,
            0.0, 1.8, -2.4,
            0.0, 1.8, -2.4,
        ])

        if self.config['training'].get('random_joint_positions', True):
            noise = np.random.uniform(-0.2, 0.2, size=12)
            joint_pos = tucked_config + noise
            for i, (lower, upper) in enumerate(self.joint_limits):
                joint_pos[i] = np.clip(joint_pos[i], lower, upper)
            for i, addr in enumerate(self.joint_qpos_addr):
                self.data.qpos[addr] = joint_pos[i]
        else:
            for i, addr in enumerate(self.joint_qpos_addr):
                self.data.qpos[addr] = tucked_config[i]

        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        for _ in range(spawn['settle_steps']):
            self.data.ctrl[:] = 0
            mujoco.mj_step(self.model, self.data)

    def _spawn_contact_stats(self):
        """Return contact count and deepest penetration (negative means overlap)."""
        deepest_penetration = 0.0
        ncon = int(self.data.ncon)
        for i in range(ncon):
            dist = float(self.data.contact[i].dist)
            if dist < deepest_penetration:
                deepest_penetration = dist
        return ncon, deepest_penetration

    def _is_valid_spawn_state(self):
        """Return True when initial contact state is not overly penetrated/stuck."""
        cfg = self.config.get('training', {})
        max_penetration = float(cfg.get('spawn_max_penetration', 0.02))
        max_contacts = int(cfg.get('spawn_max_contacts', 40))

        ncon, deepest_penetration = self._spawn_contact_stats()
        self.spawn_ncon = ncon
        self.spawn_deepest_penetration = deepest_penetration

        # Too many contacts or deep interpenetration are good proxies for stuck starts.
        if ncon > max_contacts:
            return False
        if deepest_penetration < -max_penetration:
            return False
        return True

    def _compute_recovery_score(self):
        """Compute a scalar recovery score used for progress-based early stopping."""
        uprightness = self._get_uprightness()
        height_ratio = np.clip(
            self.data.qpos[2] / self.config['robot'].get('target_height', 0.27),
            0.0,
            1.0,
        )
        return 0.7 * uprightness + 0.3 * height_ratio

    def _update_terrain_heightfield(self, roughness_level: int):
        """
        Update the height field in the simulation based on roughness level.
        Uses caching to avoid regenerating the same terrain multiple times.
        
        Args:
            roughness_level: 0-10 for terrain roughness
        """
        from environment.terrain import generate_perlin_heightfield
        
        # Check cache
        if roughness_level not in self._hfield_cache:
            # Generate new height field
            hfield_data = generate_perlin_heightfield(
                width=128,
                height=128,
                roughness_level=roughness_level,
                seed=roughness_level + self._cache_seed_offset,
                scale_factor=0.8,  # 80% amplitude variation
            )
            
            # Normalize to [-1, 1]
            hfield_normalized = (hfield_data.astype(np.float32) / 255.0 - 0.5) * 2.0
            self._hfield_cache[roughness_level] = hfield_normalized
        else:
            hfield_normalized = self._hfield_cache[roughness_level]
        
        # Update all height field geometries in the model
        # MuJoCo stores height field data as a flat array
        for hfield_id in range(len(self.model.hfield_size)):
            # Get the address in nH array where this hfield data starts
            hfield_addr = self.model.hfield_adr[hfield_id]
            hfield_size = self.model.hfield_size[hfield_id]
            
            # Only update if we have explicit height fields (not default flat plane)
            if hfield_size[0] > 0 and hfield_size[1] > 0:
                # Reshape and copy the data
                flat_size = int(hfield_size[0] * hfield_size[1])
                if flat_size == hfield_normalized.size:
                    # Copy the new height field data
                    self.model.hfield_data[hfield_addr:hfield_addr + flat_size] = hfield_normalized.flatten()


    def _get_uprightness(self):
        """Return uprightness in [0, 1], where 1 means base is upright."""
        base_quat = self.data.qpos[3:7]
        rot_matrix = self._quat_to_matrix(base_quat)
        body_z = rot_matrix @ np.array([0.0, 0.0, 1.0])
        return float(np.clip((body_z[2] + 1.0) * 0.5, 0.0, 1.0))

    def _update_termination_progress(self):
        """Track best recovery score and stagnation counter for early termination."""
        termination_cfg = self.config.get('training', {}).get('early_termination', {})
        if not termination_cfg.get('enabled', False):
            return

        current_score = self._compute_recovery_score()
        min_improvement = termination_cfg.get('min_improvement', 0.02)
        if current_score > self.best_recovery_score + min_improvement:
            self.best_recovery_score = current_score
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state - robot on its back with legs tucked"""
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # ============================================================
        # INITIAL POSITION: Robot on its back (belly up), legs tucked
        # This is the self-recovery starting position from the paper
        # ============================================================
        
        self.current_terrain = self._sample_terrain_mode()

        max_spawn_retries = int(self.config.get('training', {}).get('max_spawn_retries', 6))
        self.spawn_retry_count = 0
        self.spawn_is_valid = False
        for attempt in range(max_spawn_retries):
            self._place_robot_for_terrain(self.current_terrain)
            if self._is_valid_spawn_state():
                self.spawn_retry_count = attempt
                self.spawn_is_valid = True
                break
        if not self.spawn_is_valid:
            self.spawn_retry_count = max_spawn_retries
        
        # Reset tracking variables
        self.prev_action = np.zeros(12)
        self.current_action = np.zeros(12)  # Reset smoothed action too
        self.step_count = 0
        self.best_recovery_score = self._compute_recovery_score()
        self.steps_since_progress = 0
        
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

        self._update_termination_progress()
        
        # Check termination
        terminated = self._is_terminated(observation)
        truncated = (self.step_count + 1) >= self.config['training']['max_episode_steps']
        
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
        # Joint positions (12) - usando mapeamento correto!
        joint_positions = np.array([self.data.qpos[addr] for addr in self.joint_qpos_addr])
        
        # Joint velocities (12) - usando mapeamento correto!
        joint_velocities = np.array([self.data.qvel[addr] for addr in self.joint_qvel_addr])
        
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
        Apply PD controller to reach target joint positions.
        Uses correct joint/actuator mapping!
        Now respects per-joint torque limits from MuJoCo model.
        """
        # PD gains from config
        kp = self.config.get('controller', {}).get('kp', 40.0)
        kd = self.config.get('controller', {}).get('kd', 5.0)
        
        for i in range(12):
            # Get correct addresses from mapping
            qpos_addr = self.joint_qpos_addr[i]
            qvel_addr = self.joint_qvel_addr[i]
            actuator_id = self.actuator_ids[i]
            
            # Current state (using mapped addresses)
            current_pos = self.data.qpos[qpos_addr]
            current_vel = self.data.qvel[qvel_addr]
            
            # PD control
            error = target_positions[i] - current_pos
            torque = kp * error - kd * current_vel
            
            # Get actuator-specific torque limits from MuJoCo model
            # Hip/Thigh: ±23.7 Nm, Knee(calf): ±45.43 Nm
            ctrl_range = self.model.actuator_ctrlrange[actuator_id]
            max_torque = ctrl_range[1]  # Upper limit (symmetric)
            
            # Clip torque to actuator limits
            torque = np.clip(torque, -max_torque, max_torque)
            
            # Apply torque to CORRECT actuator
            self.data.ctrl[actuator_id] = torque
            
    def _compute_reward(self, obs, action):
        """
        Compute reward using paper's reward function
        Import from reward_manager for modularity
        """
        from src.rewards.reward_manager import RewardManager
        
        if not hasattr(self, 'reward_manager'):
            self.reward_manager = RewardManager(self.config, self.model, self.data)
        
        return self.reward_manager.compute(
            obs, action, self.prev_action, self._get_info()
        )
        
    def _is_terminated(self, obs):
        """
        Check if episode should terminate early due to clear lack of progress.
        """
        termination_cfg = self.config.get('training', {}).get('early_termination', {})
        if not termination_cfg.get('enabled', False):
            return False

        if self.step_count < termination_cfg.get('min_steps_before_check', 80):
            return False

        if self.best_recovery_score >= termination_cfg.get('success_score_threshold', 0.8):
            return False

        return self.steps_since_progress >= termination_cfg.get('patience_steps', 120)
        
    def _get_info(self):
        """Get additional info for reward computation"""
        # Base height
        base_height = self.data.qpos[2]
        
        # Foot contacts (for foot contact reward)
        # Need to check which geoms are in contact with ground
        feet_contacts = self._get_feet_contacts()
        
        # Base velocity
        base_linear_vel = self.data.qvel[0:3]

        uprightness = self._get_uprightness()
        recovery_score = self._compute_recovery_score()
        
        return {
            'base_height': base_height,
            'feet_contacts': feet_contacts,
            'base_linear_velocity': base_linear_vel,
            'uprightness': uprightness,
            'orientation_error': 1.0 - uprightness,
            'recovery_score': recovery_score,
            'best_recovery_score': self.best_recovery_score,
            'steps_since_progress': self.steps_since_progress,
            'terrain_mode': self.current_terrain,
            'current_roughness': self.current_roughness,
            'current_pitch_deg': self.current_pitch_deg,
            'spawn_retry_count': self.spawn_retry_count,
            'spawn_is_valid': float(self.spawn_is_valid),
            'spawn_contacts': self.spawn_ncon,
            'spawn_deepest_penetration': self.spawn_deepest_penetration,
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
        env = Go2Env(config)
        return env
    return _init