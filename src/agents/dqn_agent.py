"""
Factored DQN agent for MultiDiscrete action spaces.
Each action dimension gets its own Q-head sharing a common feature trunk.
Supports physics-informed reward shaping and epsilon-greedy exploration.
"""
from __future__ import annotations
import random, copy
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _QNetwork(nn.Module):
    """Shared trunk + per-dimension Q-heads."""
    def __init__(self, obs_dim: int, action_dims: List[int], hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden, d) for d in action_dims
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        h = self.trunk(x)
        return [head(h) for head in self.heads]


class DQNAgent:
    """Factored DQN with experience replay and target network."""

    def __init__(self, obs_dim: int, action_dims: List[int],
                 lr: float = 3e-4, gamma: float = 0.99,
                 eps_start: float = 1.0, eps_end: float = 0.05,
                 eps_decay: float = 0.998, buffer_size: int = 50_000,
                 batch_size: int = 64, target_update: int = 200,
                 hidden: int = 128, device: str = "cpu"):
        self.obs_dim = obs_dim
        self.action_dims = action_dims
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update
        self.device = torch.device(device)
        self.steps = 0

        self.q_net = _QNetwork(obs_dim, action_dims, hidden).to(self.device)
        self.target_net = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)

    # ---- act ----
    def act(self, obs: np.ndarray) -> np.ndarray:
        if random.random() < self.eps:
            return np.array([random.randrange(d) for d in self.action_dims])
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            qs = self.q_net(t)
            return np.array([q.argmax(dim=-1).item() for q in qs])

    # ---- store ----
    def store(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    # ---- learn ----
    def learn(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.buffer, self.batch_size)
        obs_b = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        act_b = [torch.LongTensor([b[1][i] for b in batch]).to(self.device)
                 for i in range(len(self.action_dims))]
        rew_b = torch.FloatTensor([b[2] for b in batch]).to(self.device)
        nobs_b = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        done_b = torch.FloatTensor([float(b[4]) for b in batch]).to(self.device)

        # Current Q values
        q_vals = self.q_net(obs_b)
        current_q = sum(
            q.gather(1, a.unsqueeze(1)).squeeze(1)
            for q, a in zip(q_vals, act_b)
        ) / len(self.action_dims)

        # Target Q values
        with torch.no_grad():
            nq = self.target_net(nobs_b)
            next_q = sum(q.max(dim=-1)[0] for q in nq) / len(self.action_dims)
            target_q = rew_b + self.gamma * next_q * (1 - done_b)

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Target update
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Epsilon decay
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

        return loss.item()

    def save(self, path: str):
        torch.save({"q_net": self.q_net.state_dict(), "eps": self.eps,
                     "steps": self.steps}, path)

    def load(self, path: str):
        ckpt = torch.load(path, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
        self.eps = ckpt["eps"]
        self.steps = ckpt["steps"]
