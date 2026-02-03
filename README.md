# Go2 Self-Recovery (PPO)

This repository is a **replication** of a self-righting (self-recovery) model originally developed by the **ITU** institute. It was created to support a **co-authorship** in a PhD thesis in Electrical Engineering (Automation & Control) at **UDESC**, within the **MOBI Autonomous Systems Laboratory**.

## Technical Overview
- **Robot**: Unitree Go2 quadruped
- **Simulator**: MuJoCo rigid-body physics
- **Learning Algorithm**: Proximal Policy Optimization (PPO)
- **Policy**: MLP with ReLU activations (compact architecture aligned with the reference)
- **Control**: Policy outputs joint position targets, tracked by a PD controller

## Mathematical Formulation (High-Level)
- **Observation**: 30-dimensional state vector combining joint kinematics and base motion.
- **Action**: 12-dimensional continuous joint position targets.
- **Objective**: Maximize expected return
   $$J(\pi_\theta)=\mathbb{E}\left[\sum_{t=0}^{T-1} \gamma^t r_t\right]$$
- **Reward**: Weighted sum of physically meaningful terms (height, orientation, smoothness, velocity penalties), with a curriculum component that activates near-upright states.
   $$r_t = \sum_i w_i r_i$$
- **Curriculum**: Emphasizes stabilization once the robot is sufficiently upright to avoid premature constraint on recovery dynamics.

## Deep Learning Details
- **PPO** with clipped surrogate objective for stable updates.
- **Value function** for critic-based advantage estimation (GAE).
- **Parallel environments** to increase sample throughput and reduce variance.
- **Entropy regularization** to preserve exploration during early training.

## Reproducibility Notes
- Configuration, hyperparameters, and reward weights are centralized in `config/train_config.yml`.
- Training runs are logged with TensorBoard for analysis of learning curves and evaluation metrics.

## Disclaimer
This is an **independent replication attempt** of a self-righting model from **ITU**. It is not an official release, and results may differ from the original work.
