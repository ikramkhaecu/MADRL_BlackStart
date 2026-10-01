"""Neural building blocks: MLP, dual-attention encoder (DAN, manuscript Eq. 18-19),
multi-head actors, centralised critic with physics heads (Eq. 35), QMIX mixer."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, out_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        nn.init.orthogonal_(lin.weight, gain=np.sqrt(2)); nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(sizes) - 2 or out_act:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class DAN(nn.Module):
    """Dual attention network for heterogeneous functional agents.

    environment encoder  E_i : o_i -> R^128 (two FC layers, one per agent)
    interaction encoder  f   : [pad(o_j), onehot(j)] -> R^32 (one FC + ReLU, shared)
    attention (Eq. 19)       : w_ij = softmax_j (W_q f(o_i))^T (W_k f(o_j)) / sqrt(d),
                               A_i = sum_j w_ij W_v f(o_j)      (j != i)
    output                   : h_i = [E_i(o_i), A_i]
    """

    def __init__(self, obs_dims, env_hidden=(128, 128), int_hidden=32):
        super().__init__()
        self.n, self.dmax, self.d = len(obs_dims), max(obs_dims), int_hidden
        self.env = nn.ModuleList(mlp([d, *env_hidden], out_act=True) for d in obs_dims)
        self.inter = mlp([self.dmax + self.n, int_hidden], out_act=True)
        self.wq, self.wk, self.wv = (nn.Linear(int_hidden, int_hidden, bias=False) for _ in range(3))
        self.out_dim = env_hidden[-1] + int_hidden
        self.last_attention = None

    def forward(self, obs):
        B = obs[0].shape[0]
        eye = torch.eye(self.n, device=obs[0].device)
        f = torch.stack([self.inter(torch.cat([F.pad(o, (0, self.dmax - o.shape[1])),
                                               eye[i].expand(B, -1)], 1))
                         for i, o in enumerate(obs)], 1)                     # [B, n, d]
        q, k, v = self.wq(f), self.wk(f), self.wv(f)
        att = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.d)              # [B, n, n]
        att = att.masked_fill(torch.eye(self.n, dtype=torch.bool, device=att.device), -1e9)
        w = torch.softmax(att, -1)
        self.last_attention = w.detach()
        A = torch.bmm(w, v)
        return [torch.cat([self.env[i](o), A[:, i]], 1) for i, o in enumerate(obs)]


class Actors(nn.Module):
    """Five policy heads.  mono=True: one trunk on the global state (single-agent PPO);
    otherwise one actor per functional agent on its local observation (optionally behind
    the DAN encoder); PV and WP actors share parameters when their shapes match
    ("group-based policy sharing", manuscript Table 3)."""

    def __init__(self, in_dims, n_act, hidden, mono=False, dan=None, share=True):
        super().__init__()
        self.mono, self.dan = mono, dan
        if mono:
            self.trunk = mlp([in_dims[0], *hidden], out_act=True)
            self.heads = nn.ModuleList(nn.Linear(hidden[-1], n) for n in n_act)
        else:
            feat = [dan.out_dim] * len(in_dims) if dan is not None else list(in_dims)
            nets = [mlp([feat[i], *hidden, n_act[i]]) for i in range(len(n_act))]
            if share and feat[1] == feat[2] and n_act[1] == n_act[2] and dan is None:
                nets[2] = nets[1]
            self.nets = nn.ModuleList(nets)

    def forward(self, obs):
        if self.mono:
            h = self.trunk(obs[0])
            return [hd(h) for hd in self.heads]
        x = self.dan(obs) if self.dan is not None else obs
        return [net(xi) for net, xi in zip(self.nets, x)]


class Critic(nn.Module):
    """V(s) with optional physics heads that predict the next-step bus voltages,
    settling frequency and unit powers from (s_t, a_t)  (manuscript Eq. 35)."""

    def __init__(self, in_dim, hidden, n_act_total=0, phys_dim=0):
        super().__init__()
        self.trunk = mlp([in_dim, *hidden], out_act=True)
        self.v = nn.Linear(hidden[-1], 1)
        self.phys = None
        if phys_dim:
            self.phys = mlp([hidden[-1] + n_act_total, 64, phys_dim])

    def forward(self, x):
        h = self.trunk(x)
        return self.v(h).squeeze(-1), h

    def predict(self, h, a_onehot):
        return self.phys(torch.cat([h, a_onehot], 1))


class QMixer(nn.Module):
    """Monotonic mixing network (non-negative hyper-network weights)."""

    def __init__(self, n_agents, state_dim, embed=32):
        super().__init__()
        self.n, self.embed = n_agents, embed
        self.w1 = nn.Linear(state_dim, n_agents * embed)
        self.b1 = nn.Linear(state_dim, embed)
        self.w2 = nn.Linear(state_dim, embed)
        self.b2 = mlp([state_dim, embed, 1])

    def forward(self, q, s):
        B = q.shape[0]
        w1 = torch.abs(self.w1(s)).view(B, self.n, self.embed)
        h = F.elu(torch.bmm(q.view(B, 1, self.n), w1) + self.b1(s).view(B, 1, self.embed))
        w2 = torch.abs(self.w2(s)).view(B, self.embed, 1)
        return (torch.bmm(h, w2) + self.b2(s).view(B, 1, 1)).view(B)
