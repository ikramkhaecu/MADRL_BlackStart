"""Non-learning reference policies and bounds.

  random_mask : uniform over the physics-informed mask (what the mask alone buys)
  greedy      : priority-greedy operator heuristic on the same mask
  upper_bound : relaxed bound on the load recovery ratio (adequacy + horizon)
  beam_search : best-known feasible sequence found by look-ahead search
                (lower bound on the optimum of one scenario)

v1 had none of these, so it could not be seen that a random masked policy
already reached the reported LRR.
"""
from __future__ import annotations

import copy

import numpy as np

from .blackstart_env import BlackStartEnv, KIND_OF


def random_mask_policy(env, rng):
    return [int(rng.choice(np.flatnonzero(m))) for m in env.action_masks()]


def greedy_policy(env, rng=None):
    g = env.grid
    m = env.action_masks()
    # ---- switch: prefer reaching a new unit, then the largest weighted unrestored load
    best, a_sw = -1.0, 0
    for j in np.flatnonzero(m[0][1:]):
        k = env.sw_idx[j]
        if env.bus_on[g.f[k]] and env.bus_on[g.t[k]]:
            score = 0.05                                     # loop closing: low value
        else:
            new_b, new_br = env._zone_if_closed(k)
            lw = sum(env.w_prio[i] * (1 - env.load_x[i]) * env.load_p[i]
                     for i in range(g.n_load) if g.load_bus[i] in new_b)
            un = sum(env.unit_s[u] for u in range(env.n_units)
                     if env.unit_bus[u] in new_b and not env.tripped[u])
            score = 3.0 * un + lw + 0.01
        if score > best:
            best, a_sw = score, j + 1
    # ---- load: largest priority-weighted admissible block
    a_ld, val = 0, 0.0
    for j in np.flatnonzero(m[4][1:]):
        v = env.w_prio[j] * min(env.adm[j], (1 - env.load_x[j]) * env.load_p[j])
        if v > val:
            val, a_ld = v, j + 1
    # ---- dispatch: renewables first, storage balances the rest (secondary control)
    k_cl = env.cfg["load"]["clpu_k"] if env.cfg["load"]["clpu_enabled"] else 1.0
    target = float(env._demand().sum()) * 1.03 + (val / max(env.w_prio[a_ld - 1], 1e-9) * k_cl if a_ld else 0.0)
    acts = {}
    for ag in ("pv", "wp", "ess"):
        sel = env.online & (env.unit_kind == KIND_OF[ag])
        ai = ("pv", "wp", "ess").index(ag) + 1
        if not sel.any():
            acts[ag] = 0
            continue
        lv = env.levels[ag]
        if ag == "ess":
            cap = env.unit_p[sel].sum()
        else:
            cap = ((1 - env.cfg["gfm"]["reserve_frac"]) * env.cf[sel] * env.unit_p[sel]).sum()
        want = np.clip(target / max(cap, 1e-9), lv.min(), lv.max())
        order = np.argsort(np.abs(lv - want))
        pick = next((int(j) + 1 for j in order if m[ai][j + 1]), 0)
        acts[ag] = pick
        got = lv[pick - 1] * cap if pick else (env.level[ag] * cap)
        target -= got
    return [a_sw, acts["pv"], acts["wp"], acts["ess"], a_ld]


def rollout(env, policy, seed, rng=None):
    rng = rng or np.random.default_rng(seed)
    env.reset(seed)
    done = False
    while not done:
        _, _, _, done, _ = env.step(policy(env, rng))
    return env.summary()


def evaluate_policy(system, policy, seeds, cfg=None, scenario=None):
    env = BlackStartEnv(system, cfg=cfg, use_mask=True, scenario=scenario)
    return [rollout(env, policy, s) for s in seeds]


def upper_bound(env):
    """Relaxed bound on LRR for the current scenario: adequacy (available MW incl.
    storage, no network) and horizon (one pick-up per step at the largest admissible step)."""
    c = env.cfg
    k_cl = 1.0                                               # CLPU decays -> not binding at the end
    avail = 0.0
    for k, kd in enumerate(env.unit_kind):
        avail += env.unit_p[k] if kd == "ESS" else env.cf[k] * env.unit_p[k]
    step = env.fm.max_step_mw(float(env.unit_s.sum()), 0.0, c["limits"]["rocof_soft_hz_s"],
                              c["limits"]["df_soft_hz"]) / (c["load"]["clpu_k"] if c["load"]["clpu_enabled"] else 1.0)
    return float(min(1.0, avail / k_cl / env.load_p.sum(), env.T * step / env.load_p.sum()))


def beam_search(system, seed, cfg=None, scenario=None, width=6, branch=6, rng_seed=0):
    """Best-known feasible restoration sequence of one scenario (slow; small systems)."""
    rng = np.random.default_rng(rng_seed)
    root = BlackStartEnv(system, cfg=cfg, use_mask=True, scenario=scenario)
    root.reset(seed)
    beam = [(0.0, root)]
    best = None
    while beam:
        nxt = []
        for _, e in beam:
            cands = [greedy_policy(e)] + [random_mask_policy(e, rng) for _ in range(branch - 1)]
            for a in cands:
                e2 = copy.deepcopy(e)
                _, _, _, done, _ = e2.step(a)
                score = e2.ret + e2._potential()
                if done:
                    if best is None or e2.ret > best.ret:
                        best = e2
                else:
                    nxt.append((score, e2))
        nxt.sort(key=lambda z: -z[0])
        beam = nxt[:width]
    return best.summary() if best is not None else None
