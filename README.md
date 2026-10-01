# PI-MADRL black-start restoration — repository v2

Simulation code for *"Physics-Informed Multi-Agent DRL Framework for Resilient System Restoration in
Inverter-dominated Hybrid Power Plants"*. v2 makes the code match the paper's claims with correct
electrical modelling; **all results must be regenerated with it** (see `docs/MANUSCRIPT_CHANGES.md`,
`CHANGELOG.md`). The v1 outputs that are in the submitted manuscript are kept, unchanged, in
`sim/results/legacy_v1/` for the record only.

## Layout

| path | purpose |
|---|---|
| `configs/default.yaml` | every parameter, each with a provenance tag (paper / v1 code / data / standard / PROVISIONAL) |
| `sim/networks.py` | modified IEEE 14 / 39 / 118 (pandapower `case14`, `case39`, `case118`); every modification is a tagged line of the YAML and is printed by `python -m sim.networks` |
| `sim/acpf.py` | droop-based islanded AC power flow – no slack bus; P–f and Q–V droop, unit limits, islands, sensitivities (validated against pandapower) |
| `sim/frequency.py` | aggregated VSM + droop frequency response: RoCoF, nadir, admissible load step |
| `sim/blackstart_env.py` | Dec-POMDP with five functional agents (switch, PV, WP, ESS, load); masks, look-ahead shield, protection, CLPU, sync-check, SoC, reward Eq. 20–26, PBRS with zero terminal potential |
| `sim/nets.py`, `sim/algos.py` | DAN encoder, actors, critic with physics heads, QMIX mixer; DQN, IQL, VDN, QMIX, PPO, MAPPO, PI-MAPPO |
| `sim/train.py` | component matrix (`SPECS`), train/validation/test protocol, S1–S5 |
| `sim/baselines.py` | random+mask, greedy operator, beam search, relaxed bound |
| `sim/run_experiments.py`, `sim/make_tables.py`, `figs/make_comparison_figures.py` | experiments → JSON → LaTeX tables + `claims.md` → figures |
| `tests/` | 31 tests: physics vs pandapower, frequency closed forms, network data, env contract, PBRS telescoping, masking, pipeline |
| `docs/` | `MANUSCRIPT_CHANGES.md`, `PARAMETERS.md`, `TODO_LOG.md`, `PUBLISH.md` (how to replace the default branch), `REVIEW_TODO_2026-09-18.md` |

## Install and check

```bash
pip install -r requirements.txt
python -m pytest -q                      # ~2.5 min
python -m sim.config --provisional       # the values you still have to confirm
python -m sim.networks                   # Table 2 descriptors
```

## Reproduce

```bash
# pipeline check (minutes)
python -m sim.run_experiments --quick --out sim/results_quick

# full matrix: baselines, benchmark (+S1–S5, stress test), fair comparison, ablation
python -m sim.run_experiments --seeds 10 --episodes 3000 --jobs 16
python -m sim.make_tables                # sim/results/table_*.tex + claims.md
python figs/make_comparison_figures.py   # figs/fig*.png
```

`--seeds/--episodes` default to the manuscript's Table 3 (3 seeds × 300 episodes). That budget is too
small for the v2 task (PI-MAPPO is still improving after 1000 episodes on the 14-bus system); use what
the learning curves justify and report it. One PI-MAPPO run costs ≈ 0.2 s / episode on the 14-bus,
≈ 0.4 s on the 39-bus and ≈ 2 s on the 118-bus system (single core); runs are independent, so
`--jobs` scales linearly. `--parts` and `--systems` select subsets.

Single run: `python -m sim.train --algo pi-mappo --system 14 --seed 0 --episodes 1000`.

## Reporting rules built into the code

* the reported return is always the **unshaped** environment return (`ret`); the shaped training
  signal is logged separately (`ret_shaped`);
* scenarios are random; train / validation / test seeds are disjoint; the checkpoint is selected on
  validation, numbers come from test;
* an episode whose island collapses has LRR = 0;
* every table row carries its physics-informed components (φ, mask, PBRS, L^PI, DAN);
* result files carry a label (`meta.json`), figures print it, so preliminary runs cannot be mistaken
  for final ones.

## Data availability note

The default branch of the public repository must point to this version; the prototype on `main`
(`scripts/generate_figures.py` with hard-coded multipliers) is archived under a tag. Exact commands:
`docs/PUBLISH.md`.
