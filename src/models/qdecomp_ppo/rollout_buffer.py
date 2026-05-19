"""
Rollout buffer for Decomposed PPO.
Stores per-step data and computes per-component GAE advantages.
"""

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class RolloutBatch:
    obs:       torch.Tensor   # (B, obs_dim)
    actions:   torch.Tensor   # (B, action_dim)
    log_probs: torch.Tensor   # (B, 1)
    adv_total: torch.Tensor   # (B, 1)  Σ_j A_j
    returns:   torch.Tensor   # (B, N)  discounted returns per component


class MultiRewardRolloutBuffer:
    """
    Stores a fixed-length rollout and computes:
      - GAE advantage A_j for each reward component j
      - adv_total = Σ_j A_j  (used in PPO clip objective)
      - returns_j = A_j + V_j(s)  (used in value loss)
    """

    def __init__(
        self,
        n_steps: int,
        obs_dim: int,
        action_dim: int,
        n_subagents: int,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ):
        self.n_steps     = n_steps
        self.n_subagents = n_subagents
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
        self.device      = device
        self.ptr         = 0

        self.obs       = np.zeros((n_steps, obs_dim),      dtype=np.float32)
        self.actions   = np.zeros((n_steps, action_dim),   dtype=np.float32)
        self.log_probs = np.zeros((n_steps, 1),            dtype=np.float32)
        self.rewards   = np.zeros((n_steps, n_subagents),  dtype=np.float32)
        self.values    = np.zeros((n_steps, n_subagents),  dtype=np.float32)
        self.dones     = np.zeros((n_steps, 1),            dtype=np.float32)

        # Filled by compute_returns_and_advantages
        self.adv_total = np.zeros((n_steps, 1),           dtype=np.float32)
        self.returns   = np.zeros((n_steps, n_subagents), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward_vec: np.ndarray,   # (n_subagents,) weighted rewards
        value_vec: np.ndarray,    # (n_subagents,) V_j estimates
        done: bool,
    ):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr]   = reward_vec
        self.values[self.ptr]    = value_vec
        self.dones[self.ptr]     = float(done)
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.n_steps

    def compute_returns_and_advantages(
        self, last_values: np.ndarray   # (n_subagents,) V_j(s_T)
    ):
        """
        Compute GAE per reward component, then sum for adv_total.
        last_values: value estimates at the step AFTER the buffer ends.
        """
        advantages = np.zeros((self.n_steps, self.n_subagents), dtype=np.float32)

        for j in range(self.n_subagents):
            gae = 0.0
            next_val  = last_values[j]
            next_done = 0.0

            for t in reversed(range(self.n_steps)):
                done_t    = self.dones[t, 0]
                delta     = (
                    self.rewards[t, j]
                    + self.gamma * next_val * (1 - done_t)
                    - self.values[t, j]
                )
                gae = delta + self.gamma * self.gae_lambda * (1 - done_t) * gae
                advantages[t, j] = gae
                next_val  = self.values[t, j]
                next_done = done_t

        self.returns   = advantages + self.values          # (T, N)
        self.adv_total = advantages.sum(axis=1, keepdims=True)  # (T, 1)

        # Normalize total advantage
        adv_mean = self.adv_total.mean()
        adv_std  = self.adv_total.std() + 1e-8
        self.adv_total = (self.adv_total - adv_mean) / adv_std

    def iterate_minibatches(self, batch_size: int):
        indices = np.random.permutation(self.n_steps)
        for start in range(0, self.n_steps, batch_size):
            idx = indices[start:start + batch_size]

            def _t(arr):
                return torch.as_tensor(arr[idx], device=self.device)

            yield RolloutBatch(
                obs=_t(self.obs),
                actions=_t(self.actions),
                log_probs=_t(self.log_probs),
                adv_total=_t(self.adv_total),
                returns=_t(self.returns),
            )

    def reset(self):
        self.ptr = 0
