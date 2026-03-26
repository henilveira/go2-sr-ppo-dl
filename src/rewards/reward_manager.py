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

        # Contact-jitter tracking (reset at each new episode)
        self.prev_feet_contacts = None
        self.contact_switch_count = 0
        self.contact_touchdown_count = 0
        self.foot_contact_streaks = np.zeros(4, dtype=np.float32)
        self.four_feet_contact_streak = 0
        self.stable_support_streak = 0
        
        # Standing pose reference for Go2 (in radians)
        # From MuJoCo menagerie Go2 model - typical standing configuration
        # Hip: ~0, Thigh: ~0.8 (bent forward), Calf: ~-1.5 (bent back)
        # These values should be close to the robot's natural standing pose
        self.standing_pose = np.array([
            0.0, 0.67, -1.3,   # FR: hip, thigh, calf
            0.0, 0.67, -1.3,   # FL
            0.0, 0.67, -1.3,   # RR
            0.0, 0.67, -1.3    # RL
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

        # Reset per-episode tracking at first env step.
        # In this env, first step usually arrives as step=1.
        if int(info.get('step', 0)) <= 1:
            self.prev_feet_contacts = None
            self.contact_switch_count = 0
            self.contact_touchdown_count = 0
            self.foot_contact_streaks[:] = 0.0
            self.four_feet_contact_streak = 0
            self.stable_support_streak = 0
        
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

        # Torque-efficiency terms encourage precise motion with less actuator effort.
        rewards['R_torque_efficiency'] = self._compute_torque_efficiency_reward(info)
        rewards['R_post_recovery_relax'] = self._compute_post_recovery_relax_reward(info)

        # Contact transition metrics:
        # - switches: any ON/OFF change (jitter proxy)
        # - touchdowns: only OFF->ON changes (number of contact events)
        switches, switch_ratio, touchdowns, touchdown_ratio = self._compute_contact_transition_metrics(info['feet_contacts'])
        rewards['R_jitter_contact'] = switch_ratio
        rewards['foot_contact_switches'] = float(switches)
        rewards['foot_contact_switches_total'] = float(self.contact_switch_count)
        rewards['R_touchdown_contact'] = touchdown_ratio
        rewards['foot_contact_touchdowns'] = float(touchdowns)
        rewards['foot_contact_touchdowns_total'] = float(self.contact_touchdown_count)

        # Exponential duration rewards for sustained contact.
        # This makes quick ON/OFF contact switching less rewarding than
        # maintaining stable, continuous contact.
        feet_contacts = info['feet_contacts']
        rewards['R_fc_duration_exp'], rewards['R_4fc_duration_exp'] = self._compute_contact_duration_rewards(feet_contacts)
        # Keep legacy keys for compatibility with logs/debug tooling.
        rewards['R_4fc'] = rewards['R_4fc_duration_exp']

        # Support-polygon stabilizer reward based on CoM projection and
        # largest inscribed circle approximation over the stance polygon.
        stability_reward, stability_metrics = self._compute_support_stability_reward(info)
        rewards['R_stability'] = stability_reward
        rewards.update(stability_metrics)

        # Ellipse posture reward: regularizes feet positions to stay rectangular.
        # Uses foci at midpoints of front/back paws and penalizes ellipse error.
        ellipse_reward, ellipse_metrics = self._compute_ellipse_posture_reward(info)
        rewards['R_ellipse_posture'] = ellipse_reward
        rewards.update(ellipse_metrics)
        
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
            rewards['R_fc'] = rewards['R_fc_duration_exp']
        else:
            rewards['R_h_cl'] = 0.0
            rewards['R_jp'] = 0.0
            rewards['R_fc'] = 0.0
        
        # ========== BONUS: ALIVE REWARD ==========
        # Small constant reward for surviving - encourages longer episodes
        # This helps bootstrap learning when other rewards are sparse
        rewards['R_alive'] = 0.1
        
        # ========== BONUS: UPRIGHTNESS PROGRESS ==========
        # Extra shaping reward for making progress toward upright
        # This helps guide the policy in the right direction
        z_component = self._get_body_z_in_world()
        # z_component = -1 when upside down, +1 when upright
        # Map to [0, 1]: (z + 1) / 2
        progress_reward = (z_component + 1.0) / 2.0
        rewards['R_progress'] = progress_reward
        
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
            self.weights['w8'] * rewards['R_vb'] +
            - self.weights['w9'] * rewards['R_jitter_contact'] +
            + self.weights.get('w11', 0.0) * rewards['R_4fc'] +
            + self.weights.get('w12', 0.0) * rewards['R_stability'] +
            + self.weights.get('w13', 0.0) * rewards['R_ellipse_posture'] +
            + self.weights.get('w14', 0.0) * rewards['R_torque_efficiency'] +
            + self.weights.get('w15', 0.0) * rewards['R_post_recovery_relax'] +
            # - self.weights['w10'] * rewards['R_touchdown_contact'] +
            0.05 * rewards['R_alive'] +  # Small alive bonus
            0.1 * rewards['R_progress']  # Small progress shaping
        )
        
        rewards['total'] = total_reward
        rewards['curriculum_active'] = float(self.curriculum_active)
        
        return total_reward, rewards

    def _compute_contact_transition_metrics(self, feet_contacts):
        """Return contact-transition metrics for current step.

        Returns:
            switches: ON/OFF toggles across all feet
            switch_ratio: switches normalized by number of feet
            touchdowns: OFF->ON transitions across all feet
            touchdown_ratio: touchdowns normalized by number of feet
        """
        current = np.asarray(feet_contacts, dtype=bool)
        if self.prev_feet_contacts is None:
            switches = 0
            touchdowns = 0
        else:
            switches = int(np.sum(current != self.prev_feet_contacts))
            touchdowns = int(np.sum(np.logical_and(~self.prev_feet_contacts, current)))

        self.prev_feet_contacts = current.copy()
        self.contact_switch_count += switches
        self.contact_touchdown_count += touchdowns
        switch_ratio = switches / max(len(current), 1)
        touchdown_ratio = touchdowns / max(len(current), 1)
        return switches, switch_ratio, touchdowns, touchdown_ratio

    def _compute_contact_duration_rewards(self, feet_contacts):
        """Exponential rewards based on contact persistence.

        Returns:
            r_fc_duration_exp: mean per-foot exponential persistence reward in [0, 1)
            r_4fc_duration_exp: exponential persistence reward when all 4 feet are down in [0, 1)
        """
        current = np.asarray(feet_contacts, dtype=bool)
        if current.shape[0] != 4:
            # Keep robust behavior if model contact array changes unexpectedly.
            current = np.resize(current, 4)

        # Per-foot streak in steps: increments while contact is maintained,
        # resets immediately when contact is lost.
        self.foot_contact_streaks = np.where(
            current,
            self.foot_contact_streaks + 1.0,
            0.0,
        )

        # Global streak for continuous all-4-feet contact.
        if bool(np.all(current)):
            self.four_feet_contact_streak += 1
        else:
            self.four_feet_contact_streak = 0

        duration_cfg = self.config.get('reward', {}).get('contact_duration', {})
        foot_alpha = float(duration_cfg.get('foot_alpha', 0.06))
        all_feet_alpha = float(duration_cfg.get('all_feet_alpha', 0.10))

        # exp-based saturation: fast growth early, then diminishing returns.
        foot_terms = 1.0 - np.exp(-foot_alpha * self.foot_contact_streaks)
        r_fc_duration_exp = float(np.mean(foot_terms))
        r_4fc_duration_exp = float(1.0 - np.exp(-all_feet_alpha * float(self.four_feet_contact_streak)))

        return r_fc_duration_exp, r_4fc_duration_exp

    def _compute_support_stability_reward(self, info):
        """Compute support-polygon stability reward from feet contacts and CoM projection."""
        default_metrics = {
            'support_polygon_area': 0.0,
            'support_incircle_radius': 0.0,
            'support_com_to_center_dist': 0.0,
            'support_safe_radius': 0.0,
            'support_edge_margin': -1.0,
            'support_com_inside_safe_circle': 0.0,
            'support_stability_streak': float(self.stable_support_streak),
        }

        feet_contacts = np.asarray(info.get('feet_contacts', []), dtype=bool)
        feet_positions_xy = np.asarray(info.get('feet_positions_xy', []), dtype=np.float64)
        com_world_xy = np.asarray(info.get('com_world_xy', []), dtype=np.float64)

        if feet_contacts.size == 0 or feet_positions_xy.size == 0 or com_world_xy.size != 2:
            self.stable_support_streak = 0
            default_metrics['support_stability_streak'] = 0.0
            return 0.0, default_metrics

        contact_points = []
        for i in range(min(len(feet_contacts), len(feet_positions_xy))):
            if feet_contacts[i] and np.all(np.isfinite(feet_positions_xy[i])):
                contact_points.append(feet_positions_xy[i])

        if len(contact_points) < 3:
            self.stable_support_streak = 0
            default_metrics['support_stability_streak'] = 0.0
            return 0.0, default_metrics

        points = np.asarray(contact_points, dtype=np.float64)
        hull = self._convex_hull_2d(points)
        if hull.shape[0] < 3:
            self.stable_support_streak = 0
            default_metrics['support_stability_streak'] = 0.0
            return 0.0, default_metrics

        circle_center, circle_radius = self._approx_max_inscribed_circle(hull)
        com_to_center = float(np.linalg.norm(com_world_xy - circle_center))

        stabilizer_cfg = self.config.get('reward', {}).get('stabilizer', {})
        com_radius = float(stabilizer_cfg.get('com_projection_radius', 0.05))
        area_target = float(stabilizer_cfg.get('target_polygon_area', 0.03))
        radius_target = float(stabilizer_cfg.get('target_incircle_radius', 0.09))
        streak_alpha = float(stabilizer_cfg.get('streak_alpha', 0.03))
        w_position = float(stabilizer_cfg.get('position_weight', 0.45))
        w_circle = float(stabilizer_cfg.get('incircle_weight', 0.25))
        w_area = float(stabilizer_cfg.get('area_weight', 0.20))
        w_time = float(stabilizer_cfg.get('time_weight', 0.10))

        safe_radius = max(circle_radius - com_radius, 0.0)
        edge_margin = safe_radius - com_to_center
        com_inside_safe = (safe_radius > 0.0) and (edge_margin >= 0.0)

        if bool(np.all(feet_contacts)) and com_inside_safe:
            self.stable_support_streak += 1
        else:
            self.stable_support_streak = 0

        area = self._polygon_area_2d(hull)
        position_score = np.clip(edge_margin / max(safe_radius, 1e-6), 0.0, 1.0) if safe_radius > 0.0 else 0.0
        circle_score = np.clip(circle_radius / max(radius_target, 1e-6), 0.0, 1.0)
        area_score = np.clip(area / max(area_target, 1e-6), 0.0, 1.0)
        time_score = float(1.0 - np.exp(-streak_alpha * float(self.stable_support_streak)))

        total_weight = max(w_position + w_circle + w_area + w_time, 1e-6)
        reward = (
            w_position * position_score +
            w_circle * circle_score +
            w_area * area_score +
            w_time * time_score
        ) / total_weight

        metrics = {
            'support_polygon_area': float(area),
            'support_incircle_radius': float(circle_radius),
            'support_com_to_center_dist': com_to_center,
            'support_safe_radius': float(safe_radius),
            'support_edge_margin': float(edge_margin),
            'support_com_inside_safe_circle': float(com_inside_safe),
            'support_stability_streak': float(self.stable_support_streak),
        }
        return float(reward), metrics

    def _convex_hull_2d(self, points):
        """Return convex hull vertices in counter-clockwise order."""
        if points.shape[0] <= 1:
            return points.copy()

        unique_points = np.unique(points, axis=0)
        if unique_points.shape[0] <= 1:
            return unique_points

        sorted_points = unique_points[np.lexsort((unique_points[:, 1], unique_points[:, 0]))]

        def cross(o, a, b):
            oa = a - o
            ob = b - o
            return oa[0] * ob[1] - oa[1] * ob[0]

        lower = []
        for p in sorted_points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(sorted_points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull = np.vstack((lower[:-1], upper[:-1]))
        return hull

    def _polygon_area_2d(self, polygon):
        """Compute polygon area with shoelace formula."""
        if polygon.shape[0] < 3:
            return 0.0
        x = polygon[:, 0]
        y = polygon[:, 1]
        return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def _approx_max_inscribed_circle(self, polygon):
        """Approximate maximum inscribed circle center/radius for convex polygon."""
        center = np.mean(polygon, axis=0)
        best_center = center.copy()
        best_radius = self._point_to_polygon_boundary_distance(best_center, polygon)

        bbox_span = np.max(np.ptp(polygon, axis=0))
        step = max(0.25 * bbox_span, 1e-3)
        directions = np.array([
            [1.0, 0.0], [-1.0, 0.0],
            [0.0, 1.0], [0.0, -1.0],
            [1.0, 1.0], [1.0, -1.0],
            [-1.0, 1.0], [-1.0, -1.0],
        ], dtype=np.float64)
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

        for _ in range(24):
            improved = False
            for direction in directions:
                candidate = best_center + step * direction
                if not self._point_inside_convex_polygon(candidate, polygon):
                    continue
                candidate_radius = self._point_to_polygon_boundary_distance(candidate, polygon)
                if candidate_radius > best_radius:
                    best_radius = candidate_radius
                    best_center = candidate
                    improved = True
            if not improved:
                step *= 0.5
            if step < 1e-4:
                break

        return best_center, float(best_radius)

    def _point_inside_convex_polygon(self, point, polygon):
        """Check if point is inside or on boundary of a convex polygon."""
        n = polygon.shape[0]
        if n < 3:
            return False

        eps = 1e-9
        sign = 0.0
        for i in range(n):
            a = polygon[i]
            b = polygon[(i + 1) % n]
            edge = b - a
            to_point = point - a
            cross_z = edge[0] * to_point[1] - edge[1] * to_point[0]
            if abs(cross_z) <= eps:
                continue
            if sign == 0.0:
                sign = np.sign(cross_z)
            elif np.sign(cross_z) != sign:
                return False
        return True

    def _point_to_polygon_boundary_distance(self, point, polygon):
        """Minimum Euclidean distance from a point to polygon edges."""
        min_dist = np.inf
        n = polygon.shape[0]
        for i in range(n):
            a = polygon[i]
            b = polygon[(i + 1) % n]
            dist = self._point_to_segment_distance(point, a, b)
            if dist < min_dist:
                min_dist = dist
        return float(min_dist)

    def _point_to_segment_distance(self, p, a, b):
        """Distance from point p to segment ab."""
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            return float(np.linalg.norm(p - a))
        t = float(np.dot(p - a, ab) / denom)
        t = np.clip(t, 0.0, 1.0)
        projection = a + t * ab
        return float(np.linalg.norm(p - projection))

    def _compute_ellipse_posture_reward(self, info):
        """Compute ellipse posture reward to regularize foot positions to rectangular stance.
        
        Uses ellipse with foci at midpoints of front (FL+FR) and rear (RL+RR) paws.
        Rewards contact feet that lie approximately on the ellipse circumference.
        """
        default_metrics = {
            'ellipse_posture_reward': 0.0,
            'ellipse_error_mean': 1.0,
            'ellipse_error_max': 1.0,
            'ellipse_contacts_used': 0.0,
            'ellipse_f1_f2_dist': 0.0,
            'ellipse_a': 0.0,
            'ellipse_b': 0.0,
        }

        feet_contacts = np.asarray(info.get('feet_contacts', []), dtype=bool)
        feet_positions_xy = np.asarray(info.get('feet_positions_xy', []), dtype=np.float64)

        ellipse_cfg = self.config.get('reward', {}).get('ellipse_posture', {})
        enabled = bool(ellipse_cfg.get('enabled', True))
        only_4_contacts = bool(ellipse_cfg.get('enabled_only_with_4_contacts', True))
        min_contacts = int(ellipse_cfg.get('min_contacts', 1))
        
        if not enabled:
            return 0.0, default_metrics

        contacts_count = int(np.sum(feet_contacts))

        # Legacy gate for strict 4-feet stance only.
        if only_4_contacts and contacts_count != 4:
            return 0.0, default_metrics

        # New behavior: compute reward whenever at least one foot is in contact
        # (or a configurable minimum number of contact feet).
        if contacts_count < max(min_contacts, 1):
            return 0.0, default_metrics

        if feet_positions_xy.size < 8:  # Need 4 points (2D each)
            return 0.0, default_metrics

        # Extract positions: order is [FR, FL, RR, RL]
        fr = feet_positions_xy[0]  # Front Right
        fl = feet_positions_xy[1]  # Front Left
        rr = feet_positions_xy[2]  # Rear Right
        rl = feet_positions_xy[3]  # Rear Left

        # Validate all positions are finite
        if not (np.all(np.isfinite(fr)) and np.all(np.isfinite(fl)) and
                np.all(np.isfinite(rr)) and np.all(np.isfinite(rl))):
            return 0.0, default_metrics

        # Compute foci at midpoints
        f1 = (fl + fr) / 2.0  # Front focus
        f2 = (rl + rr) / 2.0  # Rear focus

        # Distance between foci
        c = np.linalg.norm(f2 - f1)
        if c < 1e-6:
            return 0.0, default_metrics

        # Target ellipse parameters (from config)
        b_target = float(ellipse_cfg.get('b_target', 0.11))
        a_target = float(np.sqrt(c**2 + b_target**2))

        # Compute sum-of-distances to foci for each paw (ellipse definition)
        feet = [fr, fl, rr, rl]
        sum_distances = []
        for foot in feet:
            d1 = np.linalg.norm(foot - f1)
            d2 = np.linalg.norm(foot - f2)
            sum_distances.append(d1 + d2)

        # Normalized ellipse parameter s_i = sum_distances_i / (2*a_target)
        # Ideal: s_i = 1.0 (all feet on elipse)
        s_values = np.array(sum_distances) / (2.0 * a_target)
        ellipse_errors = np.abs(s_values - 1.0)

        # Reward is evaluated only on contact feet (always positive proximity reward).
        contact_indices = np.where(feet_contacts[:4])[0]
        if contact_indices.size == 0:
            return 0.0, default_metrics

        contact_errors = ellipse_errors[contact_indices]
        ke = float(ellipse_cfg.get('ke', 8.0))
        error_mean = float(np.mean(contact_errors))
        error_max = float(np.max(contact_errors))

        # Positive proximity reward: higher when closer to ellipse.
        # Scale by contact fraction so 1-2 contact stances cannot saturate this term.
        contact_fraction = float(contact_indices.size) / 4.0
        reward = float(np.exp(-ke * error_mean) * contact_fraction)

        metrics = {
            'ellipse_posture_reward': reward,
            'ellipse_error_mean': error_mean,
            'ellipse_error_max': error_max,
            'ellipse_contacts_used': float(contact_indices.size),
            'ellipse_f1_f2_dist': float(c),
            'ellipse_a': float(a_target),
            'ellipse_b': float(b_target),
        }

        return reward, metrics
    
    def _get_body_z_in_world(self):
        """Get the z-component of body's up vector in world frame"""
        import mujoco
        base_quat = self.data.qpos[3:7]
        rot_matrix = np.zeros(9)
        mujoco.mju_quat2Mat(rot_matrix, base_quat)
        rot_matrix = rot_matrix.reshape(3, 3)
        # Body's local z-axis in world frame
        body_z = rot_matrix @ np.array([0.0, 0.0, 1.0])
        # Return world z component (positive = upright)
        return body_z[2]
    
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
        """Legacy immediate-contact reward kept for backward compatibility."""
        num_contacts = sum(feet_contacts)
        return num_contacts * 0.25
    
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

    def _compute_torque_efficiency_reward(self, info):
        """Reward lower torque usage while preserving positive shaping [0,1]."""
        utilization = float(np.clip(info.get('mean_torque_utilization', 0.0), 0.0, 1.5))
        torque_cfg = self.config.get('reward', {}).get('torque', {})
        alpha = float(torque_cfg.get('efficiency_alpha', 3.0))
        return float(np.exp(-alpha * utilization))

    def _compute_post_recovery_relax_reward(self, info):
        """Extra reward for reducing torque when upright and stable."""
        stable = bool(info.get('post_recovery_stable', 0.0) > 0.5)
        if not stable:
            return 0.0

        utilization = float(np.clip(info.get('mean_torque_utilization', 0.0), 0.0, 1.0))
        torque_cfg = self.config.get('reward', {}).get('torque', {})
        hold_gain = float(torque_cfg.get('hold_relax_gain', 1.0))
        return float(np.clip(1.0 - hold_gain * utilization, 0.0, 1.0))
