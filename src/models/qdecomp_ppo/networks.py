"""
Q-Decomposition PPO networks for Go2 self-recovery.

On-policy equivalent of Q-decomposition:
  - Shared trunk extracts features once
  - N value heads: one V_j(s) per reward component
  - Decomposed advantage: A_j = r_j + γV_j(s') - V_j(s)  (via GAE)
  - Policy gradient uses Σ_j A_j as the total advantage
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

# Same reward keys as Q-Decomp SAC (7 subagents).
# R_4fc removed (always 0.0 in training — never fired).
# R_stance is synthetic (computed in agent): R_g × R_fc — the main standing signal.
REWARD_KEYS = [
    'R_h', 'R_g', 'R_jp', 'R_fc', 'R_ad', 'R_vb', 'R_stance',
]

LOG_STD_MIN = -5
LOG_STD_MAX = 0.5   # was 2 — capping at 2 saturated entropy to theoretical max


class MultiHeadPolicy(nn.Module):
    """
    Shared-trunk policy with one value head per reward component.

    Architecture:
        obs (36D)
           |
        SharedTrunk [256, 256] → ReLU
           |                  |
        Actor head         N value heads
        (mu, log_std)      V_1(s), ..., V_N(s)
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        self.action_dim = action_dim
        n = len(REWARD_KEYS)

        # Shared trunk
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        trunk_out = hidden_sizes[-1]

        # Actor
        self.mu_head      = nn.Linear(trunk_out, action_dim)
        self.log_std_head = nn.Linear(trunk_out, action_dim)

        # One value head per reward component
        self.value_heads = nn.ModuleList([
            nn.Linear(trunk_out, 1) for _ in range(n)
        ])

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)

    def get_action_and_values(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Returns:
            action    (B, action_dim)  — sampled if action is None
            log_prob  (B, 1)
            entropy   scalar
            values    list of N tensors, each (B, 1)
        """
        feat = self._features(obs)

        mu      = self.mu_head(feat)
        log_std = self.log_std_head(feat).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()
        dist    = Normal(mu, std)

        if action is None:
            x      = dist.rsample()
            action = torch.tanh(x)
        else:
            # Invert tanh to get pre-squash value
            x = torch.atanh(action.clamp(-1 + 1e-6, 1 - 1e-6))

        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy  = dist.entropy().sum(dim=-1).mean()

        values = [head(feat) for head in self.value_heads]

        return action, log_prob, entropy, values

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        feat    = self._features(obs)
        mu      = self.mu_head(feat)
        if deterministic:
            return torch.tanh(mu)
        log_std = self.log_std_head(feat).clamp(LOG_STD_MIN, LOG_STD_MAX)
        x       = Normal(mu, log_std.exp()).rsample()
        return torch.tanh(x)
