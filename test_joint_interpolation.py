"""
Test script for joint interpolation - Move robot from sitting to standing
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
from pathlib import Path
import yaml


def load_config():
    config_path = Path(__file__).parent / 'config' / 'train_config.yml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def euler_to_quat(euler):
    """Convert Euler angles (roll, pitch, yaw) to quaternion (wxyz)"""
    cy = np.cos(euler[2] * 0.5)
    sy = np.sin(euler[2] * 0.5)
    cp = np.cos(euler[1] * 0.5)
    sp = np.sin(euler[1] * 0.5)
    cr = np.cos(euler[0] * 0.5)
    sr = np.sin(euler[0] * 0.5)
    
    quat = np.zeros(4)
    quat[0] = cr * cp * cy + sr * sp * sy  # w
    quat[1] = sr * cp * cy - cr * sp * sy  # x
    quat[2] = cr * sp * cy + sr * cp * sy  # y
    quat[3] = cr * cp * sy - sr * sp * cy  # z
    return quat


def interpolate_joints(start_pos, end_pos, t):
    """
    Linear interpolation between two joint configurations
    t: 0.0 = start, 1.0 = end
    """
    t = np.clip(t, 0.0, 1.0)
    return start_pos + t * (end_pos - start_pos)


def smooth_interpolate(t):
    """
    Smooth interpolation using cubic easing (ease in-out)
    Makes movement more natural
    """
    t = np.clip(t, 0.0, 1.0)
    # Cubic ease in-out
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


def main():
    config = load_config()
    
    # Load model
    model_path = Path(__file__).parent / 'config' / config['robot']['model_path']
    model = mujoco.MjModel.from_xml_path(str(model_path))
    
    # ============================================================
    # TIMESTEPS (same as go2_env.py)
    # ============================================================
    dyn_dt = config.get('simulation', {}).get('dyn_dt', 0.001)  # Physics timestep (1ms)
    con_dt = config.get('simulation', {}).get('con_dt', 0.01)   # Control timestep (10ms)
    n_substeps = int(con_dt / dyn_dt)  # Physics steps per control step
    
    model.opt.timestep = dyn_dt
    data = mujoco.MjData(model)
    
    print(f"dyn_dt: {dyn_dt}s (physics) | con_dt: {con_dt}s (control) | n_substeps: {n_substeps}")
    
    # ============================================================
    # JOINT CONFIGURATIONS
    # Go2 joints order: [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
    #                    RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
    # ============================================================
    
    # Position: Lying flat (tucked - for self-recovery)
    LYING_CONFIG = np.array([
        0.0, 1.8, -2.4,   # FR: thigh up, calf tucked
        0.0, 1.8, -2.4,   # FL
        0.0, 1.8, -2.4,   # RR  
        0.0, 1.8, -2.4    # RL
    ])
    
    # Position: Standing (nominal stance - legs extended)
    STANDING_CONFIG = np.array([
        0.0, 0.8, -1.5,  # Pernas mais esticadas
        0.0, 0.8, -1.5,
        0.0, 0.8, -1.5,
        0.0, 0.8, -1.5
    ])
    
    # Position: Crouched/ready stance (lower than standing)
    CROUCH_CONFIG = np.array([
        0.0, 1.2, -2.0,   # FR
        0.0, 1.2, -2.0,   # FL
        0.0, 1.2, -2.0,   # RR  
        0.0, 1.2, -2.0    # RL
    ])
    
    # Position: Spread legs (for testing stability)
    SPREAD_CONFIG = np.array([
        0.3, 0.8, -1.5,   # FR: hip outward
       -0.3, 0.8, -1.5,   # FL: hip outward (opposite direction)
        0.3, 0.8, -1.5,   # RR
       -0.3, 0.8, -1.5    # RL
    ])
    
    start_config = CROUCH_CONFIG   # Começa agachado (estável no chão)
    end_config = STANDING_CONFIG   # Termina em pé
    
    initial_height = 0.25  # Para CROUCH
    initial_quat = euler_to_quat([0, 0, 0])  # Upright
    
    interpolation_duration = 2.0  # seconds
    hold_duration = 2.0  # seconds to hold final position
    
    # PD gains - aumentados para compensar gravidade e contato!
    kp = 150.0   # Position gain (era 40, muito fraco)
    kd = 8.0     # Velocity gain (damping)
    max_torque = 23.5  # Go2 limit
    
    # Initialize
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0, 0, initial_height]
    data.qpos[3:7] = initial_quat
    data.qpos[7:19] = start_config
    mujoco.mj_forward(model, data)
    
    print("=" * 50)
    print("Joint Interpolation Test")
    print("=" * 50)
    print(f"Start config: {start_config}")
    print(f"End config:   {end_config}")
    print(f"Duration: {interpolation_duration}s")
    print("=" * 50)
    
    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        sim_time = 0.0  # Track simulation time separately
        
        while viewer.is_running():
            # Use simulation time for interpolation (more accurate than wall clock)
            elapsed = sim_time
            
            # Calculate interpolation progress
            if elapsed < interpolation_duration:
                # Interpolating
                t = elapsed / interpolation_duration
                t_smooth = smooth_interpolate(t)
                target_joints = interpolate_joints(start_config, end_config, t_smooth)
                phase = "INTERPOLATING"
            elif elapsed < interpolation_duration + hold_duration:
                # Holding final position
                target_joints = end_config
                phase = "HOLDING"
            else:
                # Loop back to start
                sim_time = 0.0
                target_joints = start_config
                phase = "RESTART"
                continue
            
            # Apply PD control to reach target (this is the "control" part)
            for i in range(12):
                current_pos = data.qpos[7 + i]
                current_vel = data.qvel[6 + i]
                
                error = target_joints[i] - current_pos
                torque = kp * error - kd * current_vel
                torque = np.clip(torque, -max_torque, max_torque)
                data.ctrl[i] = torque
            
            # Run n_substeps of physics simulation (this is the "dynamics" part)
            for _ in range(n_substeps):
                mujoco.mj_step(model, data)
            
            # Advance simulation time by con_dt
            sim_time += con_dt
            
            # Update viewer
            viewer.sync()
            
            # Print status periodically
            if int(elapsed * 10) % 5 == 0:
                base_height = data.qpos[2]
                print(f"\r[{phase:12s}] t={elapsed:.1f}s | Height: {base_height:.3f}m", end="")
            
            # Sleep to maintain realtime (sleep for con_dt to match control frequency)
            time.sleep(con_dt)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
