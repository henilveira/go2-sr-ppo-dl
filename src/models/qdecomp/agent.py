"""
Q-Decomposition Soft Actor-Critic (QDecompSAC).

Implements Russell & Zimdars (2003) Q-decomposition for continuous action spaces.
Each reward component has its own pair of Q-networks; a single shared actor
acts as the arbitrator by maximising the sum of all sub-Q-values.

Connection to the paper's SARSA requirement:
  In SAC the critic target uses a' ~ π(·|s') — the action the policy *actually*
  takes in the next state — which is exactly the on-policy (SARSA) update the
  paper requires for global optimality.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

from .networks import Actor, QDecompCritic, REWARD_KEYS
from .replay_buffer import MultiRewardReplayBuffer


class QDecompSAC:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: dict,
        device: torch.device,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.reward_keys = REWARD_KEYS
        self.n_subagents = len(self.reward_keys)

        cfg = config['qdecomp']
        self.gamma = cfg.get('gamma', 0.99)
        self.tau   = cfg.get('tau', 0.005)
        hidden     = cfg.get('hidden_sizes', [256, 256])
        use_masks  = cfg.get('use_state_masks', True)
        lr         = cfg.get('learning_rate', 3e-4)
        buf_size   = cfg.get('buffer_size', 1_000_000)
        self.batch_size       = cfg.get('batch_size', 256)
        self.learning_starts  = cfg.get('learning_starts', 5_000)
        self.train_freq       = cfg.get('train_freq', 1)
        self.gradient_steps   = cfg.get('gradient_steps', 1)

        # Networks
        self.actor = Actor(obs_dim, action_dim, hidden).to(device)
        self.critics = QDecompCritic(
            self.reward_keys, action_dim, hidden,
            use_state_masks=use_masks, full_obs_dim=obs_dim,
        ).to(device)
        self.critics_target = copy.deepcopy(self.critics).to(device)
        for p in self.critics_target.parameters():
            p.requires_grad_(False)

        # Automatic entropy tuning
        target_entropy_cfg = cfg.get('target_entropy', 'auto')
        if target_entropy_cfg == 'auto':
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(target_entropy_cfg)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()

        # Optimisers
        self.actor_opt   = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critics_opt = torch.optim.Adam(self.critics.parameters(), lr=lr)
        self.alpha_opt   = torch.optim.Adam([self.log_alpha], lr=lr)

        # Replay buffer
        self.buffer = MultiRewardReplayBuffer(
            capacity=buf_size,
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_subagents=self.n_subagents,
            device=device,
        )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            action = self.actor.deterministic(obs_t)
        else:
            action, _ = self.actor.sample(obs_t)
        return action.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Store transition
    # ------------------------------------------------------------------

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward_breakdown: dict,
        next_obs: np.ndarray,
        done: bool,
    ):
        reward_vector = np.array(
            [reward_breakdown.get(k, 0.0) for k in self.reward_keys],
            dtype=np.float32,
        )
        self.buffer.store(obs, action, reward_vector, next_obs, done)

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------

    def update(self) -> dict:
        if len(self.buffer) < self.batch_size:
            return {}

        metrics = {}
        for _ in range(self.gradient_steps):
            batch = self.buffer.sample(self.batch_size)
            m = self._update_step(batch)
            for k, v in m.items():
                metrics[k] = v
        return metrics

    def _update_step(self, batch) -> dict:
        obs      = batch.obs
        action   = batch.action
        rewards  = batch.rewards   # (B, N)
        next_obs = batch.next_obs
        done     = batch.done

        alpha = self.log_alpha.exp().detach()

        # ---- Critic update (one per subagent) ----
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            q1_next, q2_next = self.critics_target(next_obs, next_action)
            # Entropy term is shared across all subagents (one actor)
            entropy_bonus = -alpha * next_log_prob  # (B, 1)

        critic_loss_total = torch.tensor(0.0, device=self.device)
        for j in range(self.n_subagents):
            r_j = rewards[:, j:j+1]  # (B, 1)
            target_j = r_j + self.gamma * (1 - done) * (
                torch.min(q1_next[j], q2_next[j]) + entropy_bonus
            )
            q1_pred = self.critics.q1_nets[j](obs, action)
            q2_pred = self.critics.q2_nets[j](obs, action)
            loss_j = F.mse_loss(q1_pred, target_j) + F.mse_loss(q2_pred, target_j)
            critic_loss_total = critic_loss_total + loss_j

        self.critics_opt.zero_grad()
        critic_loss_total.backward()
        self.critics_opt.step()

        # ---- Actor update ----
        new_action, log_prob = self.actor.sample(obs)
        q_total = self.critics.q_total(obs, new_action)  # sum of min(Q1_j, Q2_j)
        actor_loss = (alpha * log_prob - q_total).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ---- Entropy coefficient update ----
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        # ---- Soft update target networks ----
        for param, target_param in zip(
            self.critics.parameters(), self.critics_target.parameters()
        ):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'loss/critic': critic_loss_total.item(),
            'loss/actor':  actor_loss.item(),
            'loss/alpha':  alpha_loss.item(),
            'alpha':       self.alpha,
        }
