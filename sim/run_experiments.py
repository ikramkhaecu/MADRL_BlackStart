"""Run the v2 experiment matrix and write JSON results.

  baselines    random+mask, greedy operator, relaxed upper bound (+ beam search on the 14-bus)
  benchmark    7 algorithms (vanilla baselines + PI-MAPPO) x systems x seeds, S1-S5 evaluation
  fair         every baseline behind the same physics-informed mask + features as PI-MAPPO
  ablation     PI-MAPPO minus one component / MAPPO plus one component (14-bus)
  scalability  priority-load stress test of the trained PI-MAPPO policies (largest system)

Usage
  python -m sim.run_experiments --jobs 8                       # full matrix (configs/default.yaml)
  python -m sim.run_experiments --episodes 3000 --seeds 10     # recommended budget, see README
  python -m sim.run_experiments --quick                        # pipeline check (minutes)

Every finished training run is written to <out>/runs/ at once; re-running the same command after an
interruption resumes with the runs that are missing.

Every number in the manuscript must come from the JSON files written here (``make_tables``
and ``figs/make_comparison_figures.py`` read nothing else).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .baselines import beam_search, evaluate_policy, greedy_policy, random_mask_policy, upper_bound
from .blackstart_env import BlackStartEnv
from .config import load
from .stats import summarise
from .train import METRICS, SCENARIOS, SPECS, run

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = ["dqn", "iql", "vdn", "qmix", "ppo", "mappo", "pi-mappo"]
FAIR = ["dqn+pi", "iql+pi", "vdn+pi", "qmix+pi", "ppo+pi", "mappo+pi", "pi-mappo"]
ABLATION = ["mappo", "mappo+mask", "mappo+shape", "mappo+feat", "pi-mappo-nomask", "pi-mappo-nofeat",
            "pi-mappo-noshape", "pi-mappo-nophys", "pi-mappo-nodan", "pi-mappo"]
STRESS = (0, 18, 36, 54, 72)


def _task(kw):
    import torch
    torch.set_num_threads(1)
    return run(**kw)


def _aggregate(runs):
    n = min(len(r["hist"]["ret"]) for r in runs)
    hist = {k: dict(mean=np.mean([r["hist"][k][:n] for r in runs], 0).tolist(),
                    std=np.std([r["hist"][k][:n] for r in runs], 0).tolist()) for k in runs[0]["hist"]}
    out = dict(metrics={k: summarise([r["test"][k] for r in runs]) for k in METRICS},
               per_seed=[dict(seed=r["seed"], best_episode=r["best_episode"], train_s=r["train_s"],
                              **{k: r["test"][k] for k in METRICS}) for r in runs],
               fail_reasons=[r["test"].get("fail_reasons", {}) for r in runs],
               hist=hist, val_curve=np.mean([r["val_curve"] for r in runs], 0).tolist(),
               train_s=float(np.mean([r["train_s"] for r in runs])), episodes=runs[0]["episodes"],
               spec=runs[0]["spec"], n_seeds=len(runs))
    for key in ("scenarios", "stress"):
        if key in runs[0]:
            out[key] = {s: {k: summarise([r[key][s][k] for r in runs]) for k in METRICS}
                        for s in runs[0][key]}
    tr = [r["trace"] for r in runs if r.get("trace")]
    if tr:
        out["trace"] = tr[0]
    return out


def _run_path(out_dir, name, kw):
    return os.path.join(out_dir, "runs", f"{name}_{kw['system']}_{kw['algo']}_s{kw['seed']}.json")


def _task_cached(kw, out_dir, name):
    """One training run; its result is written immediately so that an interrupted matrix
    resumes where it stopped (re-running the same command skips finished runs)."""
    p = _run_path(out_dir, name, kw)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    r = _task(kw)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".tmp", "w") as f:
        json.dump(r, f, default=float)
    os.replace(p + ".tmp", p)
    return r


def _matrix(name, algos, systems, seeds, episodes, cfg, jobs, out_dir, stress_system=None):
    tasks = [dict(algo=a, system=s, seed=sd, episodes=episodes, cfg=cfg, scenarios=(name == "benchmark"),
                  trace=(sd == seeds[0]),
                  stress_levels=(STRESS if (a == "pi-mappo" and s == stress_system) else None))
             for s in systems for a in algos for sd in seeds]
    t0, done = time.time(), []
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(_task_cached, kw, out_dir, name) for kw in tasks]
            for f in as_completed(futs):
                done.append(f.result()); _progress(name, done, len(tasks), t0)
    else:
        for kw in tasks:
            done.append(_task_cached(kw, out_dir, name)); _progress(name, done, len(tasks), t0)
    res = {s: {a: _aggregate([r for r in done if r["system"] == s and r["algo"] == a]) for a in algos}
           for s in systems}
    _dump(res, out_dir, f"{name}.json")
    return res


def _progress(name, done, total, t0):
    r = done[-1]
    print(f"[{name} {len(done):3d}/{total}] {r['system']:>3}/{r['algo']:<17} s{r['seed']} "
          f"ret={r['test']['ret']:+6.2f} LRR={r['test']['lrr']:.3f} VVR={100*r['test']['vvr']:4.1f}% "
          f"FVR={100*r['test']['fvr']:4.1f}% fail={100*r['test']['failed']:3.0f}% "
          f"({r['train_s']:.0f}s, total {(time.time()-t0)/60:.1f} min)", flush=True)


def _dump(obj, out_dir, fn):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, fn), "w") as f:
        json.dump(obj, f, indent=1, default=float)
    print("wrote", os.path.join(out_dir, fn), flush=True)


def run_baselines(systems, cfg, out_dir, n_beam=5):
    sc, tr = cfg["scenario"], cfg["train"]
    seeds = [sc["test_seed_base"] + j for j in range(tr["eval_episodes"])]
    out = {}
    for s in systems:
        out[s] = {}
        for name, pol in (("random+mask", random_mask_policy), ("greedy+mask", greedy_policy)):
            ep = evaluate_policy(s, pol, seeds, cfg)
            out[s][name] = {k: summarise([e[k] for e in ep]) for k in METRICS}
        env = BlackStartEnv(s, cfg=cfg)
        ub = []
        for sd in seeds:
            env.reset(sd); ub.append(upper_bound(env))
        out[s]["upper_bound_lrr"] = summarise(ub)
        if s == "14" and n_beam:
            bs = [beam_search(s, sd, cfg) for sd in seeds[:n_beam]]
            out[s]["beam_search"] = {k: summarise([b[k] for b in bs]) for k in METRICS}
            out[s]["beam_search"]["n_scenarios"] = n_beam
        print(f"[baselines] {s}: random+mask LRR={out[s]['random+mask']['lrr'][0]:.3f} "
              f"greedy LRR={out[s]['greedy+mask']['lrr'][0]:.3f} bound={out[s]['upper_bound_lrr'][0]:.3f}", flush=True)
    _dump(out, out_dir, "baselines.json")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--systems", nargs="+", default=["14", "39", "118"])
    ap.add_argument("--parts", nargs="+", default=["baselines", "benchmark", "fair", "ablation", "scalability"])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--label", default="")
    ap.add_argument("--config", default=None)
    ap.add_argument("--bench-algos", nargs="+", default=BENCH)
    ap.add_argument("--fair-algos", nargs="+", default=FAIR)
    ap.add_argument("--ablation-algos", nargs="+", default=ABLATION)
    ap.add_argument("--probe-every", type=int, default=None)
    a = ap.parse_args()
    cfg = load(a.config)
    seeds = list(range(a.seeds)) if a.seeds else list(cfg["train"]["seeds"])
    episodes = a.episodes or cfg["train"]["episodes"]
    systems = a.systems
    if a.probe_every:
        cfg["train"]["probe_every"] = a.probe_every
    if a.quick:
        seeds, episodes, systems = [0, 1], 60, ["14"]
        cfg["train"].update(probe_every=20, probe_episodes=4, eval_episodes=10)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        commit = "unknown"
    t0 = time.time()
    meta = dict(label=a.label or ("QUICK" if a.quick else ""), seeds=seeds, episodes=episodes, systems=systems,
                parts=a.parts, commit=commit, started=time.strftime("%Y-%m-%d %H:%M:%S"), config=cfg)
    _dump(meta, a.out, "meta.json")
    if "baselines" in a.parts:
        run_baselines(systems, cfg, a.out, n_beam=2 if a.quick else 5)
    big = systems[-1]
    if "benchmark" in a.parts:
        b = _matrix("benchmark", a.bench_algos, systems, seeds, episodes, cfg, a.jobs, a.out, stress_system=big)
        if "scalability" in a.parts and "stress" in b[big].get("pi-mappo", {}):
            _dump(dict(_system=big, levels=list(STRESS), metrics=b[big]["pi-mappo"]["stress"]), a.out, "scalability.json")
    if "fair" in a.parts:
        _matrix("fair", a.fair_algos, systems[:1], seeds, episodes, cfg, a.jobs, a.out)
    if "ablation" in a.parts:
        _matrix("ablation", a.ablation_algos, systems[:1], seeds, episodes, cfg, a.jobs, a.out)
    meta["wall_min"] = (time.time() - t0) / 60
    _dump(meta, a.out, "meta.json")
    print(f"ALL DONE in {meta['wall_min']:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
