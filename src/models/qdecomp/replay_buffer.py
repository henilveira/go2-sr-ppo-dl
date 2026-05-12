"""
Replay buffer that stores a per-component reward vector alongside each transition.
Each entry in reward_vector[j] corresponds to REWARD_KEYS[j].
"""

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Batch:
    obs: torch.Tensor          # (B, obs_dim)
    action: torch.Tensor       # (B, action_dim)
    rewards: torch.Tensor      # (B, n_subagents)
    next_obs: torch.Tensor     # (B, obs_dim)
    done: torch.Tensor         # (B, 1)


class MultiRewardReplayBuffer:
    """
    Fixed-size circular buffer storing (obs, action, reward_vector, next_obs, done).
    reward_vector has one scalar per reward component (matching REWARD_KEYS order).
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        n_subagents: int,
        device: torch.device,
    ):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs      = np.zeros((capacity, obs_dim),      dtype=np.float32)
        self.action   = np.zeros((capacity, action_dim),   dtype=np.float32)
        self.rewards  = np.zeros((capacity, n_subagents),  dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim),      dtype=np.float32)
        self.done     = np.zeros((capacity, 1),            dtype=np.float32)

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward_vector: np.ndarray,   # shape (n_subagents,)
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs[self.ptr]      = obs
        self.action[self.ptr]   = action
        self.rewards[self.ptr]  = reward_vector
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr]     = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)

        def _t(arr):
            return torch.as_tensor(arr[idx], device=self.device)

        return Batch(
            obs=_t(self.obs),
            action=_t(self.action),
            rewards=_t(self.rewards),
            next_obs=_t(self.next_obs),
            done=_t(self.done),
        )

    def __len__(self) -> int:
        return self.size
