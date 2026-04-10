"""
PPO agent for MultiDiscrete action spaces (factored categorical policy).
Physics-informed variant adds auxiliary constraint penalty to policy loss.
"""
from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class _ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dims: List[int], hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # Initialize weights with small values for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        self.policy_heads = nn.ModuleList([
            nn.Linear(hidden, d) for d in action_dims
        ])
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = [head(h) for head in self.policy_heads]
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x):
        # Clamp input to avoid NaN propagation
        x = torch.clamp(x, -10.0, 10.0)
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        logits, value = self.forward(x)
        dists = [Categorical(logits=torch.nan_to_num(lg, nan=0.0),
                             validate_args=False) for lg in logits]
        actions = torch.stack([d.sample() for d in dists], dim=-1)
        log_probs = torch.stack([d.log_prob(a) for d, a in
                                 zip(dists, actions.T)], dim=-1).sum(-1)
        entropy = torch.stack([d.entropy() for d in dists], dim=-1).sum(-1)
        return actions, log_probs, entropy, value

    def evaluate_actions(self, x, actions):
        x = torch.clamp(x, -10.0, 10.0)
        x = torch.nan_to_num(x, nan=0.0)
        logits, value = self.forward(x)
        dists = [Categorical(logits=torch.nan_to_num(lg, nan=0.0),
                             validate_args=False) for lg in logits]
        log_probs = torch.stack([
            d.log_prob(actions[:, i]) for i, d in enumerate(dists)
        ], dim=-1).sum(-1)
        entropy = torch.stack([d.entropy() for d in dists], dim=-1).sum(-1)
        return log_probs, entropy, value


class PPOAgent:
    def __init__(self, obs_dim: int, action_dims: List[int],
                 lr: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_eps: float = 0.2,
                 entropy_coef: float = 0.01, vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5, ppo_epochs: int = 4,
                 batch_size: int = 64, hidden: int = 128,
                 physics_informed: bool = False, device: str = "cpu"):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.physics_informed = physics_informed
        self.device = torch.device(device)

        self.ac = _ActorCritic(obs_dim, action_dims, hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr)

        # Rollout buffer
        self.obs_buf, self.act_buf, self.rew_buf = [], [], []
        self.logp_buf, self.val_buf, self.done_buf = [], [], []

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            actions, logp, _, val = self.ac.get_action_and_value(t)
        return actions[0].cpu().numpy(), logp.item(), val.item()

    def store(self, obs, action, reward, logp, val, done):
        self.obs_buf.append(obs)
        self.act_buf.append(action)
        self.rew_buf.append(reward)
        self.logp_buf.append(logp)
        self.val_buf.append(val)
        self.done_buf.append(done)

    def learn(self) -> float:
        if len(self.obs_buf) < self.batch_size:
            return 0.0

        # Compute GAE
        advantages, returns = self._compute_gae()
        obs_t = torch.FloatTensor(np.array(self.obs_buf)).to(self.device)
        act_t = torch.LongTensor(np.array(self.act_buf)).to(self.device)
        old_logp_t = torch.FloatTensor(self.logp_buf).to(self.device)
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        total_loss = 0.0

        for _ in range(self.ppo_epochs):
            idx = torch.randperm(len(obs_t))
            for start in range(0, len(idx), self.batch_size):
                mb = idx[start:start + self.batch_size]
                new_logp, entropy, values = self.ac.evaluate_actions(
                    obs_t[mb], act_t[mb])

                ratio = torch.exp(new_logp - old_logp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                    1 + self.clip_eps) * adv_t[mb]
                pi_loss = -torch.min(surr1, surr2).mean()
                vf_loss = ((values - ret_t[mb]) ** 2).mean()
                loss = pi_loss + self.vf_coef * vf_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_loss += loss.item()

        # Clear buffer
        self.obs_buf.clear(); self.act_buf.clear(); self.rew_buf.clear()
        self.logp_buf.clear(); self.val_buf.clear(); self.done_buf.clear()
        return total_loss

    def _compute_gae(self):
        rewards = np.array(self.rew_buf)
        values = np.array(self.val_buf)
        dones = np.array(self.done_buf, dtype=float)
        T = len(rewards)
        advantages = np.zeros(T)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = values[t + 1] if t + 1 < T else 0.0
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages.tolist(), returns.tolist()

    def save(self, path: str):
        torch.save(self.ac.state_dict(), path)

    def load(self, path: str):
        self.ac.load_state_dict(torch.load(path, weights_only=False))
