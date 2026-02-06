"""Test initial position of robot"""
from environment.go2_env import Go2Env
import yaml
import numpy as np
import mujoco

with open('config/train_config.yml') as f:
    config = yaml.safe_load(f)

config['training']['max_episode_steps'] = 100

env = Go2Env(config, render_mode=None)

# Test a few resets
for i in range(3):
    obs, info = env.reset()
    
    # Get orientation
    quat = env.data.qpos[3:7]
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    mat = mat.reshape(3,3)
    body_up = mat @ np.array([0,0,1])
    
    print(f'Reset {i+1}:')
    print(f'  Height: {env.data.qpos[2]:.3f}')
    print(f'  Body up vector (world): [{body_up[0]:.2f}, {body_up[1]:.2f}, {body_up[2]:.2f}]')
    print(f'  On back (z<0): {body_up[2] < 0}')
    print(f'  Joints (thigh values): FR={env.data.qpos[8]:.2f}, FL={env.data.qpos[11]:.2f}')
    print()

env.close()
print('Done - Robot should start on back with tucked legs')
