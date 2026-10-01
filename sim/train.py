"""Training / evaluation driver.

``run(algo, system, seed, episodes)`` trains one algorithm variant on one system and returns
the learning history, the validation curve, held-out test metrics, scenario (S1-S5) metrics
and one step-by-step trace.

Protocol (changes from v1 are what the manuscript's Section 4 must now describe):
  * scenarios are random (load, PV / wind availability, SoC); train, validation and test use
    disjoint seed ranges -- v1's environment was deterministic, so its 20 "evaluation
    episodes" were one episode repeated and the checkpoint was selected on that same episode;
  * the checkpoint is selected on the *validation* seeds, the numbers reported are from the
    *test* seeds;
  * every reported return is the unshaped environment return (v1 added the shaping bonus
    5 x LRR to PI-MAPPO's evaluated reward only: 9.54 = (10 + 5) x 0.636);
  * a variant is defined by explicit component flags, so every row of every table states
    which physics-informed components the algorithm had.
"""
from __future__ import annotations

import time

import numpy as np

from .algos import PolicyAgent, ValueAgent, gae
from .blackstart_env import BlackStartEnv
from .config import load

METRICS = ("ret", "lrr", "lrr_w", "buses", "vvr", "fvr", "failed", "cs", "rocof", "nadir", "fss",
           "vmin", "vmax", "imax", "qmargin", "wasted", "ens_mwh", "t_prio", "soc_end",
           "switch_ops", "shed", "mw", "fvr_sev", "cap_bind", "s_eff_end", "p_adm_end",
           "n_isl_end")


def _spec(fam, kind=None, mono=False, mask=False, feat=False, shape=False, phys=False, dan=False):
    return dict(fam=fam, kind=kind, mono=mono, mask=mask, feat=feat, shape=shape, phys=phys, dan=dan)


PI = dict(mask=True, feat=True, shape=True, phys=True, dan=True)
SPECS = {
    # ---- benchmark: vanilla baselines vs the proposed method (manuscript Table 5 design)
    "dqn": _spec("val", "dqn", mono=True), "iql": _spec("val", "iql"),
    "vdn": _spec("val", "vdn"), "qmix": _spec("val", "qmix"),
    "ppo": _spec("ppo", mono=True), "mappo": _spec("ppo"),
    "pi-mappo": _spec("ppo", **PI),
    # ---- fair comparison: every algorithm behind the same physics-informed mask + features
    "dqn+pi": _spec("val", "dqn", mono=True, mask=True, feat=True),
    "iql+pi": _spec("val", "iql", mask=True, feat=True),
    "vdn+pi": _spec("val", "vdn", mask=True, feat=True),
    "qmix+pi": _spec("val", "qmix", mask=True, feat=True),
    "ppo+pi": _spec("ppo", mono=True, mask=True, feat=True),
    "mappo+pi": _spec("ppo", mask=True, feat=True),
    # ---- ablation: remove one component from PI-MAPPO / add one to MAPPO
    "pi-mappo-nomask": _spec("ppo", **{**PI, "mask": False}),
    "pi-mappo-nofeat": _spec("ppo", **{**PI, "feat": False}),
    "pi-mappo-noshape": _spec("ppo", **{**PI, "shape": False}),
    "pi-mappo-nophys": _spec("ppo", **{**PI, "phys": False}),
    "pi-mappo-nodan": _spec("ppo", **{**PI, "dan": False}),
    "mappo+mask": _spec("ppo", mask=True), "mappo+shape": _spec("ppo", shape=True),
    "mappo+feat": _spec("ppo", feat=True),
}
SCENARIOS = {                       # manuscript Table 4
    "S1": dict(blackstart="all"),
    "S2": dict(blackstart="ess_only"),
    "S3": dict(trip=True),
    "S4": dict(soc_init=[[0.10, 0.15], [0.85, 0.90]]),
    "S5": dict(pv_cf=[0.35, 0.35]),
}


def make_env(algo, system, cfg, scenario=None, seed=0):
    s = SPECS[algo]
    return BlackStartEnv(system, cfg=cfg, use_mask=s["mask"], use_features=s["feat"],
                         use_shaping=s["shape"], scenario=scenario, seed=seed)


def dims_of(env):
    return dict(obs_dims=env.obs_dims, state_dim=env.state_dim, n_act=env.n_act,
                phys_dim=env.phys_dim, n_bus=env.grid.n_bus, n_units=env.n_units)


def make_agent(algo, env, cfg, seed):
    s = SPECS[algo]
    if s["fam"] == "ppo":
        return PolicyAgent(dims_of(env), s, cfg, seed)
    return ValueAgent(s["kind"], dims_of(env), s, cfg, seed)


# ------------------------------------------------------------------ evaluation
def evaluate(agent, env, seeds, trace=False):
    out, tr = [], []
    for sd in seeds:
        obs, state = env.reset(int(sd))
        done = False
        while not done:
            m = env.action_masks()
            a = agent.act(obs, state, m, greedy=True)
            a = a[0] if isinstance(a, tuple) else a
            obs, state, _, done, info = env.step(a)
            if trace and sd == seeds[0]:
                tr.append(dict(t=env.t, action=[int(x) for x in a], lrr=info["lrr"], fss=info["fss"],
                               nadir=info["nadir"], rocof=info["rocof"], vmin=info["vmin"],
                               vmax=info["vmax"], picked_mw=info["picked_mw"],
                               p_unit=[float(x) for x in env.p_unit], soc=[float(x) for x in env.soc],
                               buses=info["buses"],
                               p_adm=float(env.log["p_adm"][-1]) if env.log["p_adm"] else 0.0,
                               s_eff=float(env.log["s_eff"][-1]) if env.log["s_eff"] else 0.0,
                               n_isl=int(env.log["n_isl"][-1]) if env.log["n_isl"] else 0,
                               q_unit=[float(x) for x in getattr(env, "q_unit", [])],
                               f_signed=float(info["phys"][env.grid.n_bus]
                                                                     * env.cfg["limits"]["df_soft_hz"])))
        out.append(env.summary())
    agg = {k: float(np.mean([o[k] for o in out])) for k in METRICS}
    agg["fail_reasons"] = {r: int(sum(o["fail_reason"] == r for o in out))
                           for r in {o["fail_reason"] for o in out} if r}
    return (agg, tr) if trace else agg


# -------------------------------------------------------------------- training
def _hist():
    return {k: [] for k in ("ret", "ret_shaped", "lrr", "lrr_w", "buses", "vvr", "fvr", "failed", "cs")}


def _log(hist, env):
    s = env.summary()
    for k in hist:
        hist[k].append(float(s[k]))


def train(algo, system, seed, episodes, cfg, verbose=False):
    spec, tr = SPECS[algo], cfg["train"]
    sc = cfg["scenario"]
    env = make_env(algo, system, cfg, seed=seed)
    val_env = make_env(algo, system, cfg, seed=seed)
    agent = make_agent(algo, env, cfg, seed)
    hist, val_curve, best = _hist(), [], None
    # ---- demonstration seeding: collect K greedy episodes, pre-train the actor
    if spec["fam"] == "ppo" and spec.get("mask"):
        from .baselines import greedy_policy
        demo_buf = dict(obs=[[] for _ in range(5)], state=[], act=[], masks=[[] for _ in range(5)])
        _dc = cfg.get("demo", {})
        n_demo = min(int(_dc.get("n_episodes", 30)), max(episodes // 10, 10))
        for d_ep in range(n_demo):
            d_obs, d_state = env.reset(sc["train_seed_base"] + 200_000 + d_ep)
            d_done = False
            while not d_done:
                d_m = env.action_masks()
                d_a = greedy_policy(env)
                for i in range(5):
                    demo_buf["obs"][i].append(d_obs[i]); demo_buf["masks"][i].append(d_m[i])
                demo_buf["state"].append(d_state); demo_buf["act"].append(d_a)
                d_obs, d_state, _, d_done, _ = env.step(d_a)
        agent.seed_from_demonstrations(demo_buf, epochs=int(_dc.get("bc_epochs", 5)))
        if verbose:
            print(f"  [{algo} {system} s{seed}] seeded from {n_demo} greedy demonstrations ({len(demo_buf['act'])} steps)", flush=True)
    val_seeds = [sc["val_seed_base"] + j for j in range(tr["probe_episodes"])]
    buf = None
    eps_decay = (tr["eps_end"] / tr["eps_start"]) ** (1.0 / max(episodes * tr["eps_decay_frac"], 1))
    for ep in range(episodes):
        obs, state = env.reset(sc["train_seed_base"] + seed * 100_000 + ep)
        done = False
        if spec["fam"] == "ppo":
            if buf is None:
                buf = dict(obs=[[] for _ in range(5)], state=[], act=[], logp=[], adv=[], ret=[],
                           masks=[[] for _ in range(5)], phys=[], pref=[], free=[])
            rews, vals = [], []
            while not done:
                m = env.action_masks()
                a, lp, v = agent.act(obs, state, m)
                for i in range(5):
                    buf["obs"][i].append(obs[i]); buf["masks"][i].append(m[i])
                buf["state"].append(state); buf["act"].append(a); buf["logp"].append(lp)
                obs, state, r, done, info = env.step(a)
                buf["phys"].append(info["phys"]); buf["pref"].append(info["droop"][0])
                buf["free"].append(info["droop"][1])
                rews.append(info["shaped"] if spec["shape"] else r); vals.append(v)
            adv, ret = gae(rews, vals, agent.gamma, agent.lam)
            buf["adv"] += list(adv); buf["ret"] += list(ret)
            if (ep + 1) % tr["batch_episodes"] == 0 or ep + 1 == episodes:
                agent.update(buf)
                buf = None
            agent.ent = tr["ent_start"] + (tr["ent_end"] - tr["ent_start"]) * (ep + 1) / episodes
        else:
            while not done:
                m = env.action_masks()
                a = agent.act(obs, state, m)
                nobs, nstate, r, done, info = env.step(a)
                nm = env.action_masks() if not done else [np.ones(n, bool) for n in env.n_act]
                agent.store(obs, state, a, info["shaped"] if spec["shape"] else r, nobs, nstate, done, nm)
                obs, state = nobs, nstate
                agent.update()
            agent.eps = max(tr["eps_end"], agent.eps * eps_decay)
            if (ep + 1) % tr["target_sync_episodes"] == 0:
                agent.sync_target()
        _log(hist, env)
        if (ep + 1) % tr["probe_every"] == 0 or ep + 1 == episodes:
            v = evaluate(agent, val_env, val_seeds)
            val_curve.append([ep + 1, v["ret"], v["lrr"], v["failed"]])
            if best is None or v["ret"] > best[0]:
                best = (v["ret"], agent.state_dict(), ep + 1)
            if verbose:
                print(f"  [{algo} {system} s{seed}] ep {ep+1:5d} train_ret={np.mean(hist['ret'][-tr['probe_every']:]):+.2f} "
                      f"val_ret={v['ret']:+.2f} val_lrr={v['lrr']:.3f} val_fail={v['failed']:.2f}", flush=True)
    agent.load_state_dict(best[1])
    return agent, hist, val_curve, best[2]


def run(algo, system, seed, episodes=None, cfg=None, scenarios=True, trace=False, verbose=False,
        stress_levels=None):
    cfg = cfg or load()
    tr, sc = cfg["train"], cfg["scenario"]
    episodes = episodes or tr["episodes"]
    t0 = time.time()
    agent, hist, val_curve, best_ep = train(algo, system, seed, episodes, cfg, verbose)
    train_s = time.time() - t0
    test_seeds = [sc["test_seed_base"] + j for j in range(tr["eval_episodes"])]
    res = evaluate(agent, make_env(algo, system, cfg), test_seeds, trace=trace)
    test, tr_ = res if trace else (res, None)
    out = dict(algo=algo, system=system, seed=seed, episodes=episodes, train_s=train_s,
               best_episode=best_ep, hist=hist, val_curve=val_curve, test=test, spec=SPECS[algo])
    if trace:
        out["trace"] = tr_
    if scenarios:
        out["scenarios"] = {k: evaluate(agent, make_env(algo, system, cfg, scenario=v), test_seeds)
                            for k, v in SCENARIOS.items()}
    if stress_levels:
        out["stress"] = {str(lv): evaluate(agent, make_env(algo, system, cfg,
                                                          scenario=dict(stress_loads=int(lv))), test_seeds)
                         for lv in stress_levels}
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="pi-mappo", choices=sorted(SPECS))
    ap.add_argument("--system", default="14")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=None)
    a = ap.parse_args()
    r = run(a.algo, a.system, a.seed, a.episodes, verbose=True)
    print({k: round(v, 4) for k, v in r["test"].items() if not isinstance(v, dict)})
