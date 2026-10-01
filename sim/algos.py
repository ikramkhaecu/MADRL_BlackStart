"""PI-MAPPO and the six baselines on one PyTorch backbone (manuscript Section 3.5).

  policy-gradient family (on-policy, PPO core)
    ppo       single agent on the global state, five categorical heads   (monolithic)
    mappo     five decentralised actors + centralised critic V(s)
    pi-mappo  mappo + phi^PI features + dynamic mask/shield + potential-based shaping
              + physics-informed critic loss + dual-attention encoder
  value-based family (off-policy, replay)
    dqn       one network on the global state, five heads, Q_tot = sum   (monolithic)
    iql       five independent Q-networks, independent TD on the team reward
    vdn       Q_tot = sum_i Q_i          qmix   Q_tot = monotonic mixer(Q_1..Q_5 ; s)

Differences from v1 that matter for the manuscript: separate actor / critic objectives
(Eq. 41), invalid actions masked with -inf in action selection *and* in the double-Q
bootstrap target (v1 value agents ignored masks; Eq. 40 multiplied by the mask), the
physics loss and the DAN encoder exist, heterogeneous agents (no parameter sharing across
functional roles except PV/WP).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .nets import DAN, Actors, Critic, QMixer, mlp

torch.set_num_threads(1)
NEG = -1e9


def _t(x, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def onehot_joint(act, n_act):
    return torch.cat([F.one_hot(act[:, i], n) for i, n in enumerate(n_act)], 1).float()


# ============================================================================ PPO
class PolicyAgent:
    def __init__(self, dims, spec, cfg, seed=0):
        torch.manual_seed(seed)
        tr = cfg["train"]
        self.spec, self.cfg, self.n_act = spec, cfg, dims["n_act"]
        self.mono = spec.get("mono", False)
        hid = list(tr["hidden"])
        dan = None
        if spec.get("dan") and not self.mono:
            dan = DAN(dims["obs_dims"], tuple(tr["dan_env_hidden"]), tr["dan_int_hidden"])
        in_dims = [dims["state_dim"]] * 5 if self.mono else dims["obs_dims"]
        self.actor = Actors(in_dims, self.n_act, hid, mono=self.mono, dan=dan)
        self.use_phys = bool(spec.get("phys"))
        self.critic = Critic(dims["state_dim"], hid, sum(self.n_act),
                             dims["phys_dim"] if self.use_phys else 0)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=tr["lr_policy"])
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=tr["lr_policy"])
        self.gamma, self.lam, self.clip = tr["gamma"], tr["gae_lambda"], tr["clip"]
        self.ent = tr["ent_start"]
        self.n_bus, self.n_units = dims["n_bus"], dims["n_units"]
        self.k_res = (cfg["limits"]["df_soft_hz"] / cfg["gfm"]["f_nom_hz"]) / cfg["gfm"]["droop_p"]

    def _inputs(self, obs, state):
        if self.mono:
            s = _t(state).view(1, -1) if np.ndim(state) == 1 else _t(state)
            return [s]
        return [(_t(o).view(1, -1) if np.ndim(o) == 1 else _t(o)) for o in obs]

    @torch.no_grad()
    def act(self, obs, state, masks, greedy=False):
        logits = self.actor(self._inputs(obs, state))
        a, lp = [], []
        for lg, m in zip(logits, masks):
            lg = lg.masked_fill(~_t(m, torch.bool).view(1, -1), NEG)
            d = torch.distributions.Categorical(logits=lg)
            ai = lg.argmax(-1) if greedy else d.sample()
            a.append(int(ai)); lp.append(float(d.log_prob(ai)))
        v, _ = self.critic(_t(state).view(1, -1))
        return np.array(a), np.array(lp, np.float32), float(v)

    def update(self, b):
        tr = self.cfg["train"]
        obs = [_t(np.stack(o)) for o in b["obs"]] if not self.mono else [_t(np.stack(b["state"]))]
        state = _t(np.stack(b["state"]))
        act = _t(np.stack(b["act"]), torch.long)
        old_lp = _t(np.stack(b["logp"]))
        adv, ret = _t(b["adv"]), _t(b["ret"])
        masks = [_t(np.stack(m), torch.bool) for m in b["masks"]]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        a1h = onehot_joint(act, self.n_act)
        stats = {}
        for _ in range(tr["ppo_epochs"]):
            logits = self.actor(obs)
            pol, ent = 0.0, 0.0
            for i, (lg, m) in enumerate(zip(logits, masks)):
                d = torch.distributions.Categorical(logits=lg.masked_fill(~m, NEG))
                ratio = torch.exp(d.log_prob(act[:, i]) - old_lp[:, i])
                pol = pol - torch.min(ratio * adv, torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv).mean()
                ent = ent + d.entropy().mean()
            loss_a = (pol - self.ent * ent) / len(logits)
            self.opt_a.zero_grad(); loss_a.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), tr["grad_clip_policy"])
            self.opt_a.step()
            v, h = self.critic(state)
            loss_c = tr["vf_coef"] * F.mse_loss(v, ret)
            if self.use_phys:
                tgt, pref, free = _t(np.stack(b["phys"])), _t(np.stack(b["pref"])), _t(np.stack(b["free"]))
                pred = self.critic.predict(h, a1h)
                sup = F.mse_loss(pred, tgt)
                f_hat = pred[:, self.n_bus:self.n_bus + 1]
                p_hat = pred[:, self.n_bus + 1:]
                res = (((p_hat - pref + self.k_res * f_hat) ** 2) * free).sum() / free.sum().clamp(min=1.0)
                loss_c = loss_c + tr["phys_beta_sup"] * sup + tr["phys_beta_res"] * res
                stats = dict(phys_sup=float(sup.detach()), phys_res=float(res.detach()))
            self.opt_c.zero_grad(); loss_c.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), tr["grad_clip_policy"])
            self.opt_c.step()
        return stats

    def seed_from_demonstrations(self, demo_buf, epochs=3):
        """Pre-train the actor on demonstration episodes (greedy/heuristic trajectories).
        This breaks the exploration deadlock where the agent learns to do nothing."""
        from .baselines import greedy_policy
        if not demo_buf or not demo_buf.get("act"):
            return
        obs = [__import__('torch').as_tensor(__import__('numpy').stack(o), dtype=__import__('torch').float32) for o in demo_buf["obs"]]
        act = __import__('torch').as_tensor(__import__('numpy').stack(demo_buf["act"]), dtype=__import__('torch').long)
        masks = [__import__('torch').as_tensor(__import__('numpy').stack(m), dtype=__import__('torch').bool) for m in demo_buf["masks"]]
        for _ in range(epochs):
            inps = obs if not self.mono else [__import__('torch').as_tensor(__import__('numpy').stack(demo_buf["state"]), dtype=__import__('torch').float32)]
            logits = self.actor(inps)
            loss = 0.0
            for i, (lg, m) in enumerate(zip(logits, masks)):
                lg = lg.masked_fill(~m, -1e9)
                loss = loss + __import__('torch').nn.functional.cross_entropy(lg, act[:, i])
            loss = loss / len(logits)
            self.opt_a.zero_grad(); loss.backward()
            __import__('torch').nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.opt_a.step()

    def state_dict(self):
        return copy.deepcopy((self.actor.state_dict(), self.critic.state_dict()))

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd[0]); self.critic.load_state_dict(sd[1])


def gae(rew, val, gamma, lam):
    """Finite-horizon GAE: the step index is part of the state, so the horizon is a true
    terminal (value 0) -- consistent with the zero terminal potential of the shaping."""
    T = len(rew)
    adv, last = np.zeros(T, np.float32), 0.0
    for t in reversed(range(T)):
        nxt = val[t + 1] if t + 1 < T else 0.0
        last = rew[t] + gamma * nxt - val[t] + gamma * lam * last
        adv[t] = last
    return adv, adv + np.asarray(val[:T], np.float32)


# ==================================================================== value-based
class ValueAgent:
    def __init__(self, kind, dims, spec, cfg, seed=0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        tr = cfg["train"]
        self.kind, self.cfg, self.n_act = kind, cfg, dims["n_act"]
        self.mono = kind == "dqn"
        hid = list(tr["hidden"])
        in_dims = [dims["state_dim"]] * 5 if self.mono else dims["obs_dims"]
        self.q = Actors(in_dims, self.n_act, hid, mono=self.mono, share=False)
        self.mixer = QMixer(5, dims["state_dim"]) if kind == "qmix" else None
        params = list(self.q.parameters()) + (list(self.mixer.parameters()) if self.mixer else [])
        self.opt = torch.optim.Adam(params, lr=tr["lr_value"])
        self.tq = copy.deepcopy(self.q)
        self.tmix = copy.deepcopy(self.mixer)
        self.gamma, self.eps = tr["gamma"], tr["eps_start"]
        self.cap, self.ptr, self.size = int(tr["buffer"]), 0, 0
        od, sd = dims["obs_dims"], dims["state_dim"]
        z = lambda *s, dt=np.float32: np.zeros((self.cap, *s), dt)
        self.B = dict(obs=[z(d) for d in od], nobs=[z(d) for d in od], s=z(sd), ns=z(sd),
                      a=z(5, dt=np.int64), r=z(), d=z(),
                      nm=[z(n, dt=bool) for n in self.n_act])

    def _in(self, obs, state):
        return [_t(state)] if self.mono else [_t(o) for o in obs]

    @torch.no_grad()
    def act(self, obs, state, masks, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return np.array([int(self.rng.choice(np.flatnonzero(m))) for m in masks])
        qs = self.q([x.view(1, -1) for x in self._in(obs, state)])
        return np.array([int(q.masked_fill(~_t(m, torch.bool).view(1, -1), NEG).argmax(-1))
                         for q, m in zip(qs, masks)])

    def store(self, obs, state, a, r, nobs, nstate, done, nmasks):
        i = self.ptr
        for k in range(5):
            self.B["obs"][k][i] = obs[k]; self.B["nobs"][k][i] = nobs[k]; self.B["nm"][k][i] = nmasks[k]
        self.B["s"][i], self.B["ns"][i], self.B["a"][i] = state, nstate, a
        self.B["r"][i], self.B["d"][i] = r, float(done)
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def update(self):
        bs = self.cfg["train"]["batch_size"]
        if self.size < bs:
            return
        idx = self.rng.integers(0, self.size, bs)
        B = self.B
        s, ns = _t(B["s"][idx]), _t(B["ns"][idx])
        a, r, d = _t(B["a"][idx], torch.long), _t(B["r"][idx]), _t(B["d"][idx])
        o = [s] if self.mono else [_t(x[idx]) for x in B["obs"]]
        no = [ns] if self.mono else [_t(x[idx]) for x in B["nobs"]]
        nm = [_t(m[idx], torch.bool) for m in B["nm"]]
        q = torch.stack([qi.gather(1, a[:, i:i + 1]).squeeze(1) for i, qi in enumerate(self.q(o))], 1)
        with torch.no_grad():                                   # double-Q with masked argmax
            a_star = [qi.masked_fill(~m, NEG).argmax(1, keepdim=True) for qi, m in zip(self.q(no), nm)]
            nq = torch.stack([qi.gather(1, ai).squeeze(1) for qi, ai in zip(self.tq(no), a_star)], 1)
        if self.kind == "iql":
            loss = F.mse_loss(q, (r.unsqueeze(1) + self.gamma * (1 - d).unsqueeze(1) * nq))
        elif self.kind == "qmix":
            with torch.no_grad():
                y = r + self.gamma * (1 - d) * self.tmix(nq, ns)
            loss = F.mse_loss(self.mixer(q, s), y)
        else:                                                   # dqn (heads) and vdn: additive
            loss = F.mse_loss(q.sum(1), r + self.gamma * (1 - d) * nq.sum(1))
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg["train"]["grad_clip_value"])
        self.opt.step()

    def sync_target(self):
        self.tq.load_state_dict(self.q.state_dict())
        if self.mixer is not None:
            self.tmix.load_state_dict(self.mixer.state_dict())

    def seed_from_demonstrations(self, demo_buf, epochs=3):
        """Pre-train the actor on demonstration episodes (greedy/heuristic trajectories).
        This breaks the exploration deadlock where the agent learns to do nothing."""
        from .baselines import greedy_policy
        if not demo_buf or not demo_buf.get("act"):
            return
        obs = [__import__('torch').as_tensor(__import__('numpy').stack(o), dtype=__import__('torch').float32) for o in demo_buf["obs"]]
        act = __import__('torch').as_tensor(__import__('numpy').stack(demo_buf["act"]), dtype=__import__('torch').long)
        masks = [__import__('torch').as_tensor(__import__('numpy').stack(m), dtype=__import__('torch').bool) for m in demo_buf["masks"]]
        for _ in range(epochs):
            inps = obs if not self.mono else [__import__('torch').as_tensor(__import__('numpy').stack(demo_buf["state"]), dtype=__import__('torch').float32)]
            logits = self.actor(inps)
            loss = 0.0
            for i, (lg, m) in enumerate(zip(logits, masks)):
                lg = lg.masked_fill(~m, -1e9)
                loss = loss + __import__('torch').nn.functional.cross_entropy(lg, act[:, i])
            loss = loss / len(logits)
            self.opt_a.zero_grad(); loss.backward()
            __import__('torch').nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.opt_a.step()

    def state_dict(self):
        return copy.deepcopy((self.q.state_dict(), self.mixer.state_dict() if self.mixer else None))

    def load_state_dict(self, sd):
        self.q.load_state_dict(sd[0])
        if self.mixer is not None and sd[1] is not None:
            self.mixer.load_state_dict(sd[1])
