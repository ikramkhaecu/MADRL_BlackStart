# Training on your own hardware

`train_all_systems.ipynb` runs the complete experiment matrix for the IEEE 14-, 39- and
118-bus systems and writes every file the manuscript is generated from.

## Setup

```bash
conda create -n blackstart python=3.11 -y
conda activate blackstart
pip install jupyter
cd MADRL_BlackStart_v2
jupyter notebook notebooks/train_all_systems.ipynb
```

The first cell installs numpy, scipy, torch (CPU is enough), pandapower, matplotlib, pyyaml
and pytest. A GPU gives almost nothing here: the islanded power flow, not the network update,
dominates the cost, so **cores matter and the GPU does not**. Set `JOBS` to the number of
physical cores you can spare; the matrix parallelises across runs close to linearly.

## Order

Run the cells top to bottom the first time.

1. Environment, tests, system ratings, provisional-parameter audit
2. **Reference policies and the feasibility bound** — before anything learned
3. Benchmark, one cell per system
4. Fair comparison, one cell per system
5. Ablation (14-bus) and scalability stress test (118-bus)
6. Progress check — run any time
7. Tables, `claims.md` and figures
8. Package `results_final.zip`

Section 2 comes first for a reason. On these systems a masked *random* policy already reaches
0.74 / 0.77 / 0.42 load recovery and the greedy operator rule reaches 0.79 / 0.79 / 0.82, so a
learned result is only interpretable against those and against the relaxed bound
(0.95 / 0.98 / 1.00). Without them there is no way to tell learning from the interface.

## Interrupting

Every finished run is written to `<out>/runs/` immediately and re-running a cell skips what is
already there. Stop the kernel whenever you like. The 118-bus reference policies use a runner
that checkpoints after every scenario; its cell loops until it prints `DONE`.

## Budget

`EPISODES = 600` and `SEEDS = 3` is the reported matrix. Raise episodes to 1200–3000 if you
have the compute — the PI-MAPPO validation curve was still improving at 800 on the 14-bus.
Ten seeds on `pi-mappo`, `mappo+pi` and `ppo+pi` alone would resolve the ranking among the top
three, which three seeds cannot; that is about 20 extra runs, far cheaper than ten seeds on the
whole matrix.

## Two things to check before writing anything up

**`cap_bind`** — the share of steps where the pick-up reached 95 % of the admissible block
$\Delta\mathcal{P}^{\mathrm{adm}}$. If this collapses after the first few steps, the frequency
constraint is not shaping the restoration sequence and the central contribution has no evidence
behind it.

**`fvr_sev` and FVR on the masked variants.** The shield now admits at most 0.494 Hz against a
0.5 Hz soft band, so masked variants should be at or near zero. Unmasked baselines running at
5–60 % is expected and is itself the result — it is what the mask prevents.

## What to send back

`results_final.zip` from the last cell. It holds `baselines.json`, `benchmark.json`,
`fair.json`, `ablation.json`, `scalability.json`, every `runs/*.json`, the generated `.tex`
tables, `claims.md` and the figures.
