"""
Reward Manager for Quadruped Self-Recovery
Based on Table II & III from paper: "Self-Recovery of Quadrupedal Robot Using DRL"

IMPORTANT: All rewards should be in range [0, 1] or positive for rewards,
           and the weights determine the contribution to total reward.
"""

import numpy as np


class RewardManager:
    """
    Manages reward computation based on paper's reward function
    Paper Eq. (9): r_total = Σ(w_i * r_i)
    
    Key insight from paper:
    - R_g (orientation) ranges from 0 (upright) to 2 (upside down)
    - Curriculum activates when robot is UPRIGHT (R_g close to 0)
    - Paper says "cl = 1 when R_g > 0.7" - this seems to be R_g INVERTED
      meaning "activate when orientation reward > 0.7" not the distance
    """
    
    def __init__(self, config, model, data):
        self.config = config
        self.model = model
        self.data = data
        self.weights = config['reward']['weights']
        
        # Curriculum learning flag
        self.curriculum_active = False
        
        # Standing pose reference for Go2 (in radians)
        # These are approximate standing joint angles
        self.standing_pose = np.array([
            0.0, 0.8, -1.5,   # FR: hip, thigh, calf
            0.0, 0.8, -1.5,   # FL
            0.0, 0.8, -1.5,   # RR
            0.0, 0.8, -1.5    # RL
        ])
        
    def compute(self, obs, action, prev_action, info):
        """
        Compute total reward
        
        Paper structure:
        - Always active: R_h (height), R_g (orientation), R_ad (action diff), 
                         R_v (joint vel), R_vb (base vel)
        - Curriculum (when upright): R_h again, R_jp (joint pos), R_fc (foot contact)
        
        Paper reward only (no extra shaping)
        """
        rewards = {}
        
        # ========== ALWAYS ACTIVE REWARDS ==========
        
        # R_h: Height reward - Paper Table II
        rewards['R_h'] = self._compute_height_reward(info['base_height'])
        
        # R_g: Orientation reward - Paper Table II
        # Returns value in [0, 1] where 1 = upright
        R_g_raw, R_g_normalized = self._compute_orientation_reward(obs)
        rewards['R_g'] = R_g_normalized
        rewards['R_g_raw'] = R_g_raw  # For curriculum check
        
        # R_ad: Action Difference Cost - Paper Table II
        rewards['R_ad'] = self._compute_action_difference(action, prev_action)
        
        # R_v: Joint Velocity Cost - Paper Table II
        rewards['R_v'] = self._compute_joint_velocity_cost(obs)
        
        # R_vb: Base Linear Velocity Cost - Paper Table II
        rewards['R_vb'] = self._compute_base_velocity_cost(info['base_linear_velocity'])
        
        # ========== CURRICULUM REWARDS ==========
        # Activate when robot is getting upright (R_g_normalized > threshold)
        threshold = self.config['reward']['curriculum'].get('orientation_threshold', 0.6)
        self.curriculum_active = R_g_normalized > threshold
        
        if self.curriculum_active:
            # R_h again (curriculum) - encourages maintaining height
            rewards['R_h_cl'] = self._compute_height_reward(info['base_height'])
            
            # R_jp: Joint Position Reward - Paper Table II
            rewards['R_jp'] = self._compute_joint_position_reward(obs)
            
            # R_fc: Foot Contact Reward - Paper Table II
            rewards['R_fc'] = self._compute_foot_contact_reward(info['feet_contacts'])
        else:
            rewards['R_h_cl'] = 0.0
            rewards['R_jp'] = 0.0
            rewards['R_fc'] = 0.0
        
        # ========== TOTAL REWARD ==========
        # Paper Eq. (9): weighted sum
        total_reward = (
            self.weights['w1'] * rewards['R_h'] +
            self.weights['w2'] * rewards['R_g'] +
            self.weights['w3'] * rewards['R_h_cl'] +
            self.weights['w4'] * rewards['R_jp'] +
            self.weights['w5'] * rewards['R_fc'] +
            self.weights['w6'] * rewards['R_ad'] +
            self.weights['w7'] * rewards['R_v'] +
            self.weights['w8'] * rewards['R_vb']
        )
        
        rewards['total'] = total_reward
        rewards['curriculum_active'] = float(self.curriculum_active)
        
        return total_reward, rewards
    
    def _compute_height_reward(self, height):
        """
        Paper Table II: R_h = [0,1) if h < 0.31, otherwise 1
        
        Linear interpolation from 0 to 1 as height increases to target.
        Clamp negative heights to 0.
        """
        h_target = self.config['robot'].get('target_height', 0.31)
        
        if height <= 0:
            return 0.0
        elif height < h_target:
            return height / h_target
        else:
            return 1.0
    
    def _compute_orientation_reward(self, obs):
        """
        Paper Table II: R_g = ||[0,0,-1] - θ_B||
        
        θ_B is gravity vector in body frame
        
        R_g = 0 when perfectly upright (θ_B = [0,0,-1])
        R_g = 2 when upside down (θ_B = [0,0,1])
        
        We convert to [0, 1] reward where 1 = upright
        
        IMPORTANT: We compute directly from MuJoCo data because observation
        has already been normalized and that corrupts the gravity vector!
        """
        import mujoco
        
        # Get quaternion directly from MuJoCo data (NOT from normalized obs)
        base_quat = self.data.qpos[3:7]
        
        # Convert quaternion to rotation matrix
        rot_matrix = np.zeros(9)
        mujoco.mju_quat2Mat(rot_matrix, base_quat)
        rot_matrix = rot_matrix.reshape(3, 3)
        
        # θ_B = R^(-1) * g = R^T * [0, 0, -1]
        # This gives us gravity direction in body frame
        gravity_world = np.array([0.0, 0.0, -1.0])
        theta_B = rot_matrix.T @ gravity_world
        
        # Target: gravity pointing down in body frame = [0, 0, -1]
        target = np.array([0.0, 0.0, -1.0])
        
        # Raw distance (0 to 2)
        R_g_raw = np.linalg.norm(target - theta_B)
        
        # Normalize to [0, 1] where 1 = upright
        # R_g_raw = 0 -> reward = 1 (upright)
        # R_g_raw = 2 -> reward = 0 (upside down)
        R_g_normalized = 1.0 - (R_g_raw / 2.0)
        R_g_normalized = np.clip(R_g_normalized, 0.0, 1.0)
        
        return R_g_raw, R_g_normalized
    
    def _compute_joint_position_reward(self, obs):
        """
        Paper Table II: R_jp = (1/12) * Σ(1 - (q - q̄)²)
        
        q = current joint positions
        q̄ = reference standing pose
        
        Maximum reward when joints match standing pose.
        
        We use raw values from MuJoCo, normalized by a reasonable range.
        """
        # Get raw joint positions from MuJoCo (not normalized obs)
        joint_pos = self.data.qpos[7:19].copy()
        
        # Reference standing pose for Go2
        q_ref = self.standing_pose
        
        # Compute normalized difference
        # Joint angles typically range ±3.14, so divide by that for normalization
        diff = (joint_pos - q_ref) / 3.14
        diff_squared = diff ** 2
        
        # Clip to prevent negative rewards
        terms = np.clip(1.0 - diff_squared, 0.0, 1.0)
        reward = (1.0 / 12.0) * np.sum(terms)
        
        return reward
    
    def _compute_foot_contact_reward(self, feet_contacts):
        """
        Paper Table II: R_fc = 0.25 per foot in contact; otherwise 0
        
        Maximum reward = 1.0 when all 4 feet in contact
        """
        num_contacts = sum(feet_contacts)
        reward = num_contacts * 0.25
        return reward
    
    def _compute_action_difference(self, action, prev_action):
        """
        Paper Table II: R_ad = (1/12) * Σ(1 - (a_t - a_{t-1})²)
        
        Penalizes large action changes (promotes smooth control).
        Actions are in [-1, 1], so diff is in [-2, 2], diff² in [0, 4].
        
        We clip to ensure positive reward.
        """
        diff = action - prev_action
        diff_squared = diff ** 2
        
        # Clip terms to [0, 1] to prevent negative rewards
        terms = np.clip(1.0 - diff_squared, 0.0, 1.0)
        reward = (1.0 / 12.0) * np.sum(terms)
        
        return reward
    
    def _compute_joint_velocity_cost(self, obs):
        """
        Paper Table II: R_v = (1/12) * Σ(1 - q̇²)
        
        Penalizes high joint velocities.
        We use raw values from MuJoCo, normalized by max expected velocity.
        """
        # Get raw joint velocities from MuJoCo
        joint_vel = self.data.qvel[6:18].copy()
        
        # Normalize by max expected velocity (30 rad/s is pretty fast)
        max_vel = 30.0
        vel_normalized = joint_vel / max_vel
        vel_squared = vel_normalized ** 2
        
        # Clip terms to [0, 1]
        terms = np.clip(1.0 - vel_squared, 0.0, 1.0)
        reward = (1.0 / 12.0) * np.sum(terms)
        
        return reward
    
    def _compute_base_velocity_cost(self, base_linear_velocity):
        """
        Paper Table II: R_vb = e^(-2(v̄_b - v_b)²)
        
        v̄_b = target velocity = 0 (want robot to stay in place)
        v_b = current base velocity magnitude
        
        Returns ~1 when still, decays exponentially with velocity.
        """
        velocity_magnitude = np.linalg.norm(base_linear_velocity)
        
        # R_vb = exp(-2 * v²) since target = 0
        reward = np.exp(-2.0 * velocity_magnitude ** 2)
        
        return reward
