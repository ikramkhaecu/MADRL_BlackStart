# Changelog

## v2.1.0 (2026-09-21) – author decisions applied
- IEEE 123-node feeder **replaced by the IEEE 118-bus system** (`case118`); the unit ratings of the
  manuscript's Table 2 (10×30 MW PV, 9×25 MW WP, 10×60 MWh ESS) now apply without rescaling.
  `sim/feeder123.py` and `sim/data/ieee123/` removed.
- Confirmed and re-tagged in `configs/default.yaml`: switchable transformers, partial sequenced
  pick-up, the 39-bus modifications, horizons 15/30/60 × 5/2.5/1.25 min.
- Look-ahead shield now also checks the thermal limit (Eq. 4).
- Speed: masks cached per step, vectorised admittance assembly and set-points, sensitivities only
  when needed (118-bus episode 2.4 s → 1.6 s; 14-bus 93 ms → 67 ms; identical results).

## v2.0.0 (2026-09-18) – code brought in line with the manuscript, results to be regenerated

### Fixed (affects every reported number)
- PI-MAPPO was evaluated with the shaping bonus (5·LRR) included, baselines without → all returns are
  now unshaped; PBRS uses γ and a zero terminal potential (policy-invariant).
- Deterministic environment + checkpoint selection on the evaluated episode → random scenarios,
  disjoint train / validation / test seeds.
- Infinite bus (`ext_grid`) removed: islands are formed by the GFM units (droop-based AC power flow);
  frequency model moved to the online-MVA base with droop and a real nadir.
- `cap_frac` LRR ceiling per system removed; ratings now follow Table 2 (14/39-bus) or are rescaled
  with a stated rule (123-node).
- Value-based agents ignored action masks → masked ε-greedy and masked double-Q targets.

### Added (claimed in the manuscript, absent in v1)
- Five functional agents with line/transformer switching and fleet dispatch actions; φ^PI features;
  DAN encoder; physics-informed critic loss; scenarios S1–S5; SoC in MWh with zones; priorities.
- (v2.0 only; superseded in v2.1) real IEEE 123-node feeder from OpenDSS data instead of the synthetic random feeder.
- Look-ahead shield, joint-action frequency projection, CLPU, transformer inrush rule, sync-check,
  spinning reserve, UFLS, protection trips.
- Baselines (random+mask, greedy, beam search, relaxed bound), fair-comparison table, five-component
  ablation, bootstrap CI / IQM, provenance-tagged configuration, 31 tests.

### Removed
- `sim/powergrid.py` (superseded by `acpf.py`, `frequency.py`, `networks.py`).
- v1 result JSONs moved to `sim/results/legacy_v1/` (superseded, kept for transparency).
