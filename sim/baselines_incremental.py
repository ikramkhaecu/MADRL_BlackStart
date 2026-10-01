"""Resumable reference-policy runner.

``run_experiments --parts baselines`` evaluates every reference policy over all test
scenarios in one call.  On the 118-bus system (99 loads, 45 steps) that exceeds the
wall-clock budget of a single interactive call, so this module does the same work
seed-by-seed and checkpoints after every scenario.  Re-running the same command picks
up where it stopped; the final ``baselines.json`` is byte-identical in structure to the
one ``run_experiments`` writes.

    python -m sim.baselines_incremental --system 118 --out sim/results/v25_bl118
"""
from __future__ import annotations

import argparse
import json
import os

from .baselines import (beam_search, greedy_policy, random_mask_policy, rollout,
                        upper_bound)
from .blackstart_env import BlackStartEnv
from .config import load
from .stats import summarise
from .train import METRICS

POLICIES = {"random+mask": random_mask_policy, "greedy+mask": greedy_policy}


def _ckpt_path(out_dir, system):
    return os.path.join(out_dir, f"_baselines_partial_{system}.json")


def _load_ckpt(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"policies": {}, "bound": [], "beam": []}


def _save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def run(system, out_dir, cfg=None, n_beam=0, budget_s=240.0):
    """Evaluate every reference policy on ``system``, checkpointing after each scenario."""
    import time
    t0 = time.time()
    cfg = cfg or load()
    sc, tr = cfg["scenario"], cfg["train"]
    seeds = [sc["test_seed_base"] + j for j in range(tr["eval_episodes"])]
    path = _ckpt_path(out_dir, system)
    ck = _load_ckpt(path)

    import numpy as np
    for name, pol in POLICIES.items():
        done = ck["policies"].setdefault(name, [])
        env = BlackStartEnv(system, cfg=cfg)
        while len(done) < len(seeds):
            if time.time() - t0 > budget_s:
                _save(ck, path)
                print(f"[{system}] {name}: {len(done)}/{len(seeds)} scenarios "
                      f"(paused, re-run to continue)", flush=True)
                return None
            sd = seeds[len(done)]
            ep = rollout(env, pol, sd, np.random.default_rng(sd))
            done.append({k: float(ep[k]) for k in METRICS})
        print(f"[{system}] {name}: complete, LRR="
              f"{np.mean([e['lrr'] for e in done]):.3f}", flush=True)
        _save(ck, path)

    env = BlackStartEnv(system, cfg=cfg)
    while len(ck["bound"]) < len(seeds):
        if time.time() - t0 > budget_s:
            _save(ck, path)
            print(f"[{system}] bound: {len(ck['bound'])}/{len(seeds)} (paused)", flush=True)
            return None
        env.reset(seeds[len(ck["bound"])])
        ck["bound"].append(float(upper_bound(env)))
    _save(ck, path)

    while len(ck["beam"]) < n_beam:
        if time.time() - t0 > budget_s:
            _save(ck, path)
            print(f"[{system}] beam: {len(ck['beam'])}/{n_beam} (paused)", flush=True)
            return None
        b = beam_search(system, seeds[len(ck["beam"])], cfg)
        ck["beam"].append({k: float(b[k]) for k in METRICS})
    _save(ck, path)

    # ---- assemble in the same shape run_experiments writes
    res = {system: {}}
    for name in POLICIES:
        eps = ck["policies"][name]
        res[system][name] = {k: summarise([e[k] for e in eps]) for k in METRICS}
    res[system]["upper_bound_lrr"] = summarise(ck["bound"])
    if ck["beam"]:
        res[system]["beam_search"] = {k: summarise([b[k] for b in ck["beam"]])
                                      for k in METRICS}
        res[system]["beam_search"]["n_scenarios"] = len(ck["beam"])

    merged_path = os.path.join(out_dir, "baselines.json")
    merged = {}
    if os.path.exists(merged_path):
        with open(merged_path) as f:
            merged = json.load(f)
    merged.update(res)
    _save(merged, merged_path)
    print(f"[{system}] DONE  random+mask={res[system]['random+mask']['lrr'][0]:.3f} "
          f"greedy={res[system]['greedy+mask']['lrr'][0]:.3f} "
          f"bound={res[system]['upper_bound_lrr'][0]:.3f}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beam", type=int, default=0)
    ap.add_argument("--budget", type=float, default=240.0)
    a = ap.parse_args()
    run(a.system, a.out, n_beam=a.beam, budget_s=a.budget)
