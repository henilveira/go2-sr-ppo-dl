"""
Q-Decomposition networks for Go2 self-recovery.

Each reward component gets its own pair of Q-networks (double-Q trick).
A single shared Actor serves as the arbitrator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Reward component names (must match keys in reward_manager.compute() breakdown)
REWARD_KEYS = [
    'R_h', 'R_g', 'R_h_cl', 'R_jp', 'R_fc',
    'R_ad', 'R_v', 'R_vb', 'R_alive', 'R_progress',
]

# Observation indices each subagent cares about (state partitioning).
# Obs layout (36D):
#   [0:12]  joint_pos
#   [12:24] joint_vel
#   [24:27] base_orientation (gravity in body frame)
#   [27:30] base_angular_vel
#   [30:33] com_pos_base
#   [33:36] com_vel_base
STATE_MASKS: dict[str, list[int]] = {
    'R_h':       [24, 25, 26] + list(range(30, 33)),   # orientation + com_pos (height proxy)
    'R_g':       [24, 25, 26],
    'R_h_cl':    [24, 25, 26] + list(range(30, 33)),
    'R_jp':      list(range(0, 12)),
    'R_fc':      list(range(0, 12)) + [24, 25, 26],
    'R_ad':      list(range(0, 36)),
    'R_v':       list(range(12, 24)),
    'R_vb':      [27, 28, 29] + list(range(33, 36)),   # ang_vel + com_vel
    'R_alive':   list(range(0, 36)),
    'R_progress':[24, 25, 26],
}

LOG_STD_MIN = -5
LOG_STD_MAX = 2


def _trunk(input_dim: int, hidden_sizes: list[int]) -> nn.Sequential:
    """Feature trunk: hidden layers with ReLU, no output projection."""
    layers: list[nn.Module] = []
    in_dim = input_dim
    for h in hidden_sizes:
        layers += [nn.Linear(in_dim, h), nn.ReLU()]
        in_dim = h
    return nn.Sequential(*layers)


def _mlp(input_dim: int, hidden_sizes: list[int], output_dim: int) -> nn.Sequential:
    """MLP with ReLU activations; last layer is Linear with no activation."""
    layers: list[nn.Module] = []
    in_dim = input_dim
    for h in hidden_sizes:
        layers += [nn.Linear(in_dim, h), nn.ReLU()]
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Shared stochastic policy (arbitrator).
    Outputs a tanh-squashed Gaussian action distribution.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        self.net = _trunk(obs_dim, hidden_sizes)
        self.mu_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, obs: torch.Tensor):
        x = self.net(obs)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log-probability (with tanh correction)."""
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        x = dist.rsample()
        action = torch.tanh(x)
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _ = self.forward(obs)
        return torch.tanh(mu)


class SubQNetwork(nn.Module):
    """
    Q-network for a single reward component.
    Accepts a (possibly masked) observation concatenated with the action.
    """

    def __init__(self, obs_indices: list[int], action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        self.register_buffer('obs_indices', torch.tensor(obs_indices, dtype=torch.long))
        input_dim = len(obs_indices) + action_dim
        self.net = _mlp(input_dim, hidden_sizes, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs_part = obs[:, self.obs_indices]
        x = torch.cat([obs_part, action], dim=-1)
        return self.net(x)


class QDecompCritic(nn.Module):
    """
    Container for all sub-Q-networks.
    Holds two Q-networks per subagent (double-Q trick for stability).
    """

    def __init__(
        self,
        reward_keys: list[str],
        action_dim: int,
        hidden_sizes: list[int],
        use_state_masks: bool = True,
        full_obs_dim: int = 36,
    ):
        super().__init__()
        self.reward_keys = reward_keys
        self.use_state_masks = use_state_masks

        q1_nets = []
        q2_nets = []
        for key in reward_keys:
            indices = STATE_MASKS[key] if use_state_masks else list(range(full_obs_dim))
            q1_nets.append(SubQNetwork(indices, action_dim, hidden_sizes))
            q2_nets.append(SubQNetwork(indices, action_dim, hidden_sizes))

        self.q1_nets = nn.ModuleList(q1_nets)
        self.q2_nets = nn.ModuleList(q2_nets)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        q1_vals = [net(obs, action) for net in self.q1_nets]
        q2_vals = [net(obs, action) for net in self.q2_nets]
        return q1_vals, q2_vals

    def q_total(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Sum of min(Q1_j, Q2_j) over all subagents — used for actor update."""
        q1_vals, q2_vals = self.forward(obs, action)
        total = sum(torch.min(q1, q2) for q1, q2 in zip(q1_vals, q2_vals))
        return total
