# TODO log — repository v2 rebuild (saved 2026-09-18, updated 2026-09-21)

Status: [x] done and tested · [~] done, needs author confirmation / full-budget run · [ ] open (owner)

## A. Repository integrity
- [x] A1 Snapshot v1 (`master`, commit 0bf954a) in git; move v1 result JSONs to `sim/results/legacy_v1/`.
- [x] A2 Delete `sim/powergrid.py`; rewrite `feeder123.py`, `blackstart_env.py`, `algos.py`, `train.py`, `run_experiments.py`, `make_tables.py`, `figs/make_comparison_figures.py`.
- [x] A3 `CHANGELOG.md` states what was wrong in v1 and what changed.
- [~] A4 Decision taken (2026-09-21): replace `main` with v2. Commands prepared in `docs/PUBLISH.md` (tags `archive/main-prototype`, `v1-submitted`, force-push `v2:main`).
- [ ] A4b (author) Run `docs/PUBLISH.md` – cannot be done from the assistant's side (no GitHub access).
- [ ] A5 (author) Tag the commit used for the revised manuscript; put the tag in the Data-availability statement.

## B. Configuration and provenance
- [x] B1 `configs/default.yaml`: every parameter with a tag [paper] / [code-v1] / [code-main] / [data] / [standard] / [PROVISIONAL].
- [x] B2 `python -m sim.config --provenance|--provisional|--markdown`; `docs/PARAMETERS.md` generated.
- [x] B3a Confirmed by the author 2026-09-21 and re-tagged `[confirmed:author 2026-09-21]`: transformers switchable; partial sequenced pick-up (+ block interval); dt = 5 min; 39-bus load scale 0.1656, 70 % shunt compensation, 7 split loads, 17-line rule, unit buses; horizons 15/30/60 × 5/2.5/1.25 min; IEEE-118 instead of IEEE-123.
- [ ] B3b (author) Still open (`python -m sim.config --provisional`, 37 rows): v_trip, df_trip_hz, rocof_trip_hz_s, t_lpf_s, pf_rated, reserve_frac, ESS duration_h, soc_init, min_pickup_frac, CLPU k/τ, taps_nominal, inrush_pu, surrogate ratings, scenario ranges (pv_cf, wp_cf, cf_walk_std), w_line, w_pi, adequacy_min, PBRS w_n, dispatch levels, phys-loss betas, probe/eval episodes, 14-bus shunt out; 118-bus: load scale 0.3007, 70 % shunt compensation, shunts out, 86-line rule, placement rule.
- [ ] B4 (author) Verify the IEEE 1547-2018 Table 5 sync limits (0.1 Hz, 3 %, 10°) against the standard.

## C. Test systems
- [x] C1 IEEE-14: ESS@0 (85 MWh → 42.5 MW), PV@1 55 MW, WP@2 70 MW; 15 lines + 5 transformers switchable; surrogate ratings 1.5 × base-case flow; capacitor out; taps nominal.
- [~] C2 IEEE-39: 11 units on the original generator buses, loads ×0.1656 (1035.7 MW), 7 largest loads split → 28, 70 % line shunt compensation (310.8 Mvar left), 17 largest-charging lines + 3 network transformers switchable, T = 30 × 2.5 min. Runs, zero collapses under masked reference policies; **all modifications provisional**.
- [x] C3 IEEE-123 feeder removed (author decision 2026-09-21); `sim/feeder123.py`, `sim/data/ieee123/` deleted.
- [~] C4 IEEE-118 (`case118`): paper Table-2 ratings unchanged (10×30 MW PV, 9×25 MW WP, 10×60 MWh/30 MW ESS = 825 MW / 916.7 MVA); loads ×0.3007 → 1275.6 MW (99 loads); 70 % line shunt compensation (302.7 Mvar left); 14 bus shunts out; units on the 29 largest-capability generator buses; 86 largest-charging lines + 12 network transformers switchable (98); T = 60 × 1.25 min. At t = 0 the ten ESS have soft-energised their zones: 53 buses, 9 islands, 19 units online. Masked reference policies: 0 collapses, 0 violations. **Scaling / compensation / placement / switch rule provisional.**
- [ ] C4b (author) Table 2 third column → IEEE 118-bus: 99 loads (not 95), 98 switchable branches (not 38); replace "123-bus" at the manuscript lines listed in `docs/MANUSCRIPT_CHANGES.md` §4.
- [x] C5 `python -m sim.networks` / `describe()` → Table 2 (`table_env.tex`).

## D. Electrical model
- [x] D1 Droop-based islanded AC power flow (no slack), P–f and Q–V droop in the equations, P/Q/current limits by clamping, per-island solve, flat-start retry, Jacobian sensitivities; max |ΔV| vs pandapower 1.5e-9 pu.
- [x] D2 Frequency model: VSM + droop + LPF on the online-MVA base; RoCoF, nadir (1.058 × settling), admissible block; limit-aware droop balance predictor (exact on break-points).
- [x] D3 Switching physics: zone energisation, line charging vs var margin, Ferranti/tap factor, transformer inrush rule with soft-energisation window, sync-check relay (loop and island merge).
- [x] D4 Load model: partial/binary pick-up, sequenced blocks, CLPU, priorities, spinning-reserve rule, staged UFLS on renewable drops.
- [x] D5 ESS: MWh-based SoC (Eq. 7), zones, pre-stop limits, P̄ = E/duration; PV/WP deloading reserve and availability random walk.
- [x] D6 Protection: soft bands → VVR/FVR; trip bands, inrush over-current, inadequate generation, PF failure → collapse (LRR = 0, r_done = −1).
- [x] D7 Scenarios S1–S5 + priority-load stress test (0/18/36/54/72).
- [ ] D8 (optional, author) Time-domain replay of a learned sequence (multi-GFM classical model or ANDES REGF1/REGCV1) for one electrical validation figure – not implemented in v2; `fig14` shows quasi-steady traces + per-event RoCoF/nadir only.
- [x] D9 Obsolete (123-node feeder removed; all three systems are balanced transmission cases).

## E. MDP and learning code
- [x] E1 Five functional agents; n_act 14-bus [21,7,7,10,12], 39-bus [21,7,7,10,29], 118-bus [99,7,7,10,100].
- [x] E2 Observations (local per agent + system block), global state, φ^PI (8 features, switchable).
- [x] E3 Masks (Eq. 27–32) + look-ahead shield (V, f, thermal) + joint dispatch projection; `feas` flags stay in the observation when the mask is off.
- [x] E4 Reward Eq. 20–26 (+SoC, +wasted), horizon-normalised penalties, collapse forfeits restored load; `last_terms` logged.
- [x] E5 PBRS with γ and zero terminal potential; unshaped `ret` and shaped `ret_shaped` separated; telescoping test.
- [x] E6 DAN (env encoders 128-128, shared interaction encoder 32, q/k/v attention, no self-attention); actors only.
- [x] E7 PolicyAgent (PPO mono, MAPPO, PI-MAPPO): separate actor/critic optimisers, masked categorical, linear entropy anneal, finite-horizon GAE, physics heads + droop residual.
- [x] E8 ValueAgent (DQN mono, IQL, VDN, QMIX): masked ε-greedy, masked double-Q target, numpy replay, target sync.
- [x] E9 `SPECS` component matrix: benchmark, `+pi` fair variants, remove-one / add-one ablations.
- [x] E10 Train/validation/test protocol, checkpoint on validation, traces, S1–S5, stress levels.
- [ ] E11 (author) Decide the final training budget from the learning curves (PI-MAPPO still improves after 1000 episodes; paper-Table-3 lr 2e-3 / batch 4 is fast but noisy – validation curve oscillates) and update Table 3.

## F. Experiments, tables, figures
- [x] F1 `run_experiments`: baselines, benchmark (+scenarios, +stress), fair, ablation, scalability; `--jobs`, `--quick`, `--parts`, `--systems`, per-part algorithm filters, `meta.json` with config snapshot + label.
- [x] F2 `stats.py`: bootstrap CI, IQM; `make_tables.py`: 9 table types + `claims.md`.
- [x] F3 Figures 5, 9, 9b, 10, 11, 12, 13, 14 with seed bands and run label.
- [~] F4 Preliminary sandbox run (1 CPU) on the final v2.1 code: reference policies on 14/39/118; 14-bus benchmark 3 seeds × 600 episodes; reduced ablation and fair tables (2 seeds) → `sim/results/preliminary_*` (label printed on every figure); see `docs/MANUSCRIPT_CHANGES.md` §7 for what finished.
- [ ] F5 (author) Full run: `python -m sim.run_experiments --seeds 10 --episodes <budget> --jobs <cores>`; then `make_tables`, figures; paste `claims.md` sentences into the manuscript.
- [ ] F6 (author) Beam search currently only on the 14-bus (5 scenarios); extend if an optimality gap is wanted on the larger systems.

## G. Tests (31, all passing on the final v2.1 code, 2026-09-21)
- [x] G1 `test_physics.py` (6): PF vs pandapower, droop laws, clamping, islands, frequency closed forms, network descriptors.
- [x] G2 `test_env.py` (12): reproducibility, shapes, masked policies inside limits, unmasked pick-up trips protection, bookkeeping + PBRS telescoping, φ flag, 39/118 run, S1/S3/S4/S5 run.
- [x] G3 `test_algos.py` (13): smoke training ×7, masked agents never select invalid actions, QMIX monotone, DAN attention, SPECS completeness, end-to-end pipeline.

## H. Manuscript
- [x] H1 `docs/MANUSCRIPT_CHANGES.md`: section-by-section replacements, deleted claims, table/figure mapping.
- [ ] H2 (author) Apply H1; rewrite Sections 4–5 from regenerated outputs; update Fig. 2 (DAN → actors only) and Table 2/3.
- [ ] H3 (author) Citation fixes P6.x of `docs/REVIEW_TODO_2026-09-18.md`.
