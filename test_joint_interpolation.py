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
    # JOINT/ACTUATOR MAPPING (mesmo que go2_env.py)
    # ============================================================
    legs = ['FR', 'FL', 'RR', 'RL']
    links = ['hip', 'thigh', 'calf']
    
    joint_names = []
    actuator_names = []
    for leg in legs:
        for link in links:
            joint_names.append(f"{leg}_{link}_joint")
            actuator_names.append(f"{leg}_{link}")
    
    # Map joints -> qpos/qvel addresses
    joint_qpos_addr = []
    joint_qvel_addr = []
    for name in joint_names:
        joint_id = model.joint(name).id
        joint_qpos_addr.append(model.jnt_qposadr[joint_id])
        joint_qvel_addr.append(model.jnt_dofadr[joint_id])
    
    # Map actuators -> ctrl indices
    actuator_ids = []
    for name in actuator_names:
        actuator_ids.append(model.actuator(name).id)
    
    print("\nJOINT/ACTUATOR MAPPING:")
    for i, (jname, aname) in enumerate(zip(joint_names, actuator_names)):
        qpos = joint_qpos_addr[i]
        ctrl = actuator_ids[i]
        print(f"  [{i:2d}] {jname:20s} -> qpos[{qpos}], ctrl[{ctrl}]")
    
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
    warmup_duration = 0.5  # seconds to stabilize before interpolating
    
    # PD gains - balanceados
    kp = 80.0    # Position gain (não muito alto para evitar explosão)
    kd = 5.0     # Velocity gain (damping)
    max_torque = 23.5  # Go2 limit
    
    # Initialize
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0, 0, initial_height]
    data.qpos[3:7] = initial_quat
    
    # Set joint positions using correct mapping
    for i, addr in enumerate(joint_qpos_addr):
        data.qpos[addr] = start_config[i]
    
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
        sim_time = 0.0
        actual_start_config = None  # Será capturado após warmup
        
        while viewer.is_running():
            elapsed = sim_time
            
            # ============================================================
            # FASES: WARMUP -> INTERPOLATING -> HOLDING -> RESTART
            # ============================================================
            if elapsed < warmup_duration:
                # WARMUP: Manter posição atual (onde o robô está agora)
                # Deixa estabilizar antes de começar
                target_joints = np.array([data.qpos[addr] for addr in joint_qpos_addr])
                phase = "WARMUP"
                current_kp = kp * (elapsed / warmup_duration)  # Ramp up gradualmente
                
            elif elapsed < warmup_duration + interpolation_duration:
                # Captura posição atual como start no primeiro frame de interpolação
                if actual_start_config is None:
                    actual_start_config = np.array([data.qpos[addr] for addr in joint_qpos_addr])
                    print(f"\n\nActual start captured: {actual_start_config}")
                
                # INTERPOLATING: Mover de posição atual para end
                t = (elapsed - warmup_duration) / interpolation_duration
                t_smooth = smooth_interpolate(t)
                target_joints = interpolate_joints(actual_start_config, end_config, t_smooth)
                phase = "INTERPOLATING"
                current_kp = kp
                
            elif elapsed < warmup_duration + interpolation_duration + hold_duration:
                # HOLDING: Manter posição final
                target_joints = end_config
                phase = "HOLDING"
                current_kp = kp
                
            else:
                # RESTART: Volta pro início
                sim_time = 0.0
                actual_start_config = None  # Reset para capturar novamente
                # Reset positions
                for i, addr in enumerate(joint_qpos_addr):
                    data.qpos[addr] = start_config[i]
                mujoco.mj_forward(model, data)
                phase = "RESTART"
                continue
            
            # Apply PD control to reach target (using correct mapping!)
            for i in range(12):
                qpos_addr = joint_qpos_addr[i]
                qvel_addr = joint_qvel_addr[i]
                actuator_id = actuator_ids[i]
                
                current_pos = data.qpos[qpos_addr]
                current_vel = data.qvel[qvel_addr]
                
                error = target_joints[i] - current_pos
                torque = current_kp * error - kd * current_vel
                torque = np.clip(torque, -max_torque, max_torque)
                data.ctrl[actuator_id] = torque
            
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
                max_ctrl = np.max(np.abs(data.ctrl))
                print(f"\r[{phase:12s}] t={elapsed:.1f}s | Height: {base_height:.3f}m | Max torque: {max_ctrl:.1f}", end="")
            
            # Sleep to maintain realtime (sleep for con_dt to match control frequency)
            time.sleep(con_dt)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
