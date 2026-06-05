"""
Decomposed PPO agent.

On-policy equivalent of Q-decomposition:
  - N value heads V_j(s), one per reward component
  - GAE computed independently per component
  - adv_total = Σ_j A_j used in the PPO clip objective
  - value_loss = Σ_j MSE(V_j(s), returns_j)
"""

import numpy as np
import torch
import torch.nn.functional as F

from .networks import MultiHeadPolicy, REWARD_KEYS
from .rollout_buffer import MultiRewardRolloutBuffer


class DecompPPO:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: dict,
        device: torch.device,
    ):
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.device     = device
        self.reward_keys = REWARD_KEYS
        self.n_subagents = len(REWARD_KEYS)

        cfg = config['qdecomp_ppo']
        self.n_steps       = cfg.get('n_steps',       2048)
        self.batch_size    = cfg.get('batch_size',     256)
        self.n_epochs      = cfg.get('n_epochs',       10)
        self.gamma         = cfg.get('gamma',          0.99)
        self.gae_lambda    = cfg.get('gae_lambda',     0.95)
        self.clip_range    = cfg.get('clip_range',     0.2)
        self.vf_coef       = cfg.get('vf_coef',        0.5)
        self.ent_coef      = cfg.get('ent_coef',       1e-3)
        self.max_grad_norm = cfg.get('max_grad_norm',  0.5)
        hidden             = cfg.get('hidden_sizes',   [256, 256])

        # Weighted contributions (same as SAC version).
        # R_stance is synthetic (R_g × R_fc) — computed in collect_rollout.
        w = config['reward']['weights']
        self._reward_scale = {
            'R_h':     w.get('w1',  0.30),
            'R_g':     w.get('w2',  0.30),
            'R_jp':    w.get('w4',  0.60),
            'R_fc':    w.get('w5',  0.10),
            'R_ad':    w.get('w6',  0.05),
            'R_vb':    w.get('w8',  0.05),
            # Synthetic "standing" signal: fires when BOTH upright AND
            # foot-contacting. Replaces the dead R_4fc.
            'R_stance': 0.50,
        }

        # R_stance hold-streak (see SAC agent for rationale): rewards SUSTAINING
        # the stand, resets on roll/fall so repeated rolling cannot farm it.
        self._stance_streak = 0
        self._stance_alpha  = cfg.get('stance_streak_alpha', 0.05)
        self._stance_gate   = cfg.get('stance_gate', 0.6)

        self.policy    = MultiHeadPolicy(obs_dim, action_dim, hidden).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.get('learning_rate', 3e-4))

        self.buffer = MultiRewardRolloutBuffer(
            n_steps=self.n_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_subagents=self.n_subagents,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=device,
        )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Returns (action, log_prob, value_vec)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, log_prob, _, values = self.policy.get_action_and_values(obs_t)
        if deterministic:
            action = self.policy.predict(obs_t, deterministic=True)
        value_vec = np.array([v.item() for v in values], dtype=np.float32)
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value_vec,
        )

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def _compute_component(self, key: str, breakdown: dict) -> float:
        """Return the weighted reward for a single subagent component."""
        if key == 'R_stance':
            val = self._stance_value(breakdown)
        else:
            val = breakdown.get(key, 0.0)
        return self._reward_scale[key] * val

    def _stance_value(self, breakdown: dict) -> float:
        """
        Synthetic 'stand and HOLD' reward: upright AND foot-contacting, growing
        with sustained duration. Rolling/falling resets the streak.
        """
        r_g  = breakdown.get('R_g', 0.0)
        r_fc = breakdown.get('R_fc', 0.0)
        if r_g > self._stance_gate and r_fc > 0.0:
            self._stance_streak += 1
        else:
            self._stance_streak = 0
        hold = 1.0 - np.exp(-self._stance_alpha * self._stance_streak)
        return r_g * r_fc * hold

    # ------------------------------------------------------------------
    # Collect rollout
    # ------------------------------------------------------------------

    def collect_rollout(self, env) -> dict:
        """Collect self.n_steps transitions. Returns last obs after rollout."""
        self.buffer.reset()
        obs, _ = env.reset()
        ep_rewards = []
        ep_reward  = 0.0

        for _ in range(self.n_steps):
            action, log_prob, value_vec = self.select_action(obs)
            next_obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            breakdown  = info.get('reward_breakdown', {})
            reward_vec = np.array(
                [self._compute_component(k, breakdown) for k in self.reward_keys],
                dtype=np.float32,
            )
            ep_reward += sum(reward_vec)

            self.buffer.add(obs, action, log_prob, reward_vec, value_vec, done)
            obs = next_obs if not done else env.reset()[0]

            if done:
                self._stance_streak = 0  # reset hold-streak at episode boundary
                ep_rewards.append(ep_reward)
                ep_reward = 0.0

        # Bootstrap value at the end of rollout
        _, _, last_values = self.select_action(obs)
        self.buffer.compute_returns_and_advantages(last_values)

        return {
            'rollout/mean_ep_reward': float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            'rollout/n_episodes':     len(ep_rewards),
        }

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self) -> dict:
        total_loss_actor  = 0.0
        total_loss_value  = 0.0
        total_entropy     = 0.0
        n_updates         = 0

        for _ in range(self.n_epochs):
            for batch in self.buffer.iterate_minibatches(self.batch_size):
                _, new_log_prob, entropy, new_values = self.policy.get_action_and_values(
                    batch.obs, batch.actions
                )

                # PPO clip objective
                log_ratio  = new_log_prob - batch.log_probs
                ratio      = log_ratio.exp()
                adv        = batch.adv_total

                loss_actor = -torch.min(
                    ratio * adv,
                    ratio.clamp(1 - self.clip_range, 1 + self.clip_range) * adv,
                ).mean()

                # Value loss: sum over all components
                loss_value = sum(
                    F.mse_loss(new_values[j], batch.returns[:, j:j+1])
                    for j in range(self.n_subagents)
                )

                loss = loss_actor + self.vf_coef * loss_value - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss_actor += loss_actor.item()
                total_loss_value += loss_value.item()
                total_entropy    += entropy.item()
                n_updates        += 1

        n = max(n_updates, 1)
        return {
            'loss/actor':  total_loss_actor / n,
            'loss/value':  total_loss_value / n,
            'loss/entropy': total_entropy   / n,
        }
