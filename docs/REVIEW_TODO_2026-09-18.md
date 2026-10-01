# Manuscript 4 (PI-MADRL black start) — revision TODO log

- Reviewed: `Fourth_Manuscript_20_08_2026-5.pdf` (27 pp.) — `L###` = manuscript line number
- Repo checked: `github.com/ikramkhaecu/MADRL_BlackStart` @ `a065148` (cloned 2026-09-18)
- Data checked programmatically: pandapower 3.5.4 (`case14`, `case39`), OpenDSS reference `IEEE123Loads.DSS` / `IEEE123Master.dss`, ANDES 2.0.0 model list
- Not checked: reference list against DOIs; any private code not in the repo
- Bibliographic suggestions below are from memory — verify details before citing

Raw verification output:

```
case14: buses=14 lines=15 trafos=5 loads=11 P_load=259.0 MW Q_load=73.5 Mvar
case39: buses=39 lines=35 trafos=11 loads=21 P_load=6254.2 MW Q_load=1387.1 Mvar
IEEE 123-node feeder (OpenDSS): 91 load elements, 3490 kW, 1920 kvar, basekv=4.16
Paper Table 2 on 123-bus: PV 300 MW + WP 225 MW = 150x the reference feeder load

Released env (_build_ieee14_blackstart):
  ext_grid: bus 0, vm_pu 1.06, in_service True      <- infinite bus is the reference
  gen rows 0 | sgen rows 5 | trafos 5 (no switches, always in service) | switches 15 (= lines)
  all lines closed, all loads on, ALL IBR P = 0  ->  ext_grid P = 278.2 MW, Q = 111.7 Mvar
  random policy, 200 eps x 15 steps: LRR mean 0.110 max 0.661 | buses mean 3.21 max 5
  buses reachable from bus 0 via lines only: [0, 1, 2, 3, 4]   (paper reports 12.0)
  ext_grid P at episode end: mean -46.1 MW, min -164.4, max +73.1

grep -i over src/ scripts/ (hits): grid2op 0 (labels only) | qmix 0 | vdn 0 | iql 0 |
  potential 0 | case39 0 | swing 0 | inertia 0 | slack 0 | ext_grid 0
ANDES 2.0.0: REGCV1, REGCV2 (VSG), REGF1 (droop), REGF2 (VSM), REGF3 (dVOC); ieee14/ieee39 dynamic cases bundled
```

---

## P0 — Blocking: reproducibility / integrity (before anyone else opens the link at L660)

- [ ] P0.1 `scripts/generate_figures.py::fig_algorithm_benchmark` (L399–441 of that file): PI-MAPPO bars = `dqn_lrr*1.3`, `dqn_rew*1.5`, `dqn_bus*1.4`; "Random" hard-coded (0.02, −25, 1.5). Delete; plot only from logged runs. Manuscript Table 5 has PI-MAPPO 9.54 = exactly 1.5 × 6.36 (VDN/QMIX) — a reviewer will connect the two.
- [ ] P0.2 Delete `fig_ev_distribution()` (random EV data, unrelated to this paper) from this repo.
- [ ] P0.3 Push the code that produced Tables 5–7 / Figs 4–8: environments for 14/39/123-bus, IQL, VDN, QMIX, MAPPO, PI-MAPPO, DAN, masks, PBRS, physics loss, configs. Repo currently has one 14-bus env + DQN + PPO only.
- [ ] P0.4 Commit raw per-episode logs (CSV/JSON) for every algorithm × network × seed, plus one script that regenerates every table and figure from those logs.
- [ ] P0.5 State the simulator truthfully. Repo imports pandapower directly; paper says Grid2Op 1.9 (L387, L390, Alg. 2 l.16, Fig. 2 label, Table 5 caption `sim/`, L632). Either port to Grid2Op or change the text and cite pandapower (Thurner et al., IEEE TPWRS 2018).
- [ ] P0.6 Make L407 true (`scripts/train.py` "unified across algorithms" — it trains DQN and PPO, no seeds) and Table 5 caption true (`sim/` does not exist).
- [ ] P0.7 After P1 fixes, re-derive every number in abstract, §5.3 and conclusion from the logs.
- [ ] P0.8 Add `requirements.txt`/lock file, seeds, README with exact commands. Remove committed `__pycache__` (`cpython-314` — paper says Python 3.10).
- [ ] P0.9 The released env cannot reproduce the paper (max 5 buses vs 12.0 reported). Whatever produced 12.0 must be what is released.

## P1 — Evaluation methodology

- [ ] P1.1 Report the unshaped return (Eq. 26) for ALL algorithms; shaped return is a training signal only. Evidence (Fig. 7): MAPPO+mask → LRR 0.636, R 6.36; PI-MAPPO → LRR 0.636, R 9.54; MAPPO+shaping → LRR 0.614 but R 8.10.
- [ ] P1.2 Add an algorithm × component matrix (φ^PI features, mask, PBRS, physics loss, DAN). Resolve: Alg. 2 l.5–9 (all algorithms get features + mask) vs §3.7 L335–337 (PI components only on proposed algorithm) vs Table 5 MAPPO = 0.420 = unmasked MAPPO of Fig. 7.
- [ ] P1.3 Add baselines: (a) uniform-random + mask; (b) priority-greedy heuristic + mask; (c) MILP/MISOCP (or exhaustive/beam search) optimum on the 14-bus → report optimality gap.
- [ ] P1.4 Explain why every masked algorithm lands on exactly 0.636 (= 7/11 loads) with std 0.000 while Fig. 4b shows untrained baselines at ≈0.69. It is not a physical ceiling. Report MW restored and per-priority-class restoration, not only the ratio.
- [ ] P1.5 Explain Fig. 4b: baseline LRR falls (~0.69 → ~0.635; PPO/MAPPO to ~0.47–0.52) while reward rises; L426–435 says LRR "improves". Check train-vs-eval logging and whether an always-on penalty (r^PI, Eq. 24) rewards restoring less. Also explain reward ≈ 0.5 at episode 0 with LRR ≈ 0.69 and |Δf| ≈ 0.24 Hz < limit.
- [ ] P1.6 Explain the flat PI-MAPPO curves from episode 0 (Fig. 4a–d). Show what learning adds beyond the mask (compare against P1.3a).
- [ ] P1.7 Training budget: 300 × 15 = 4,500 env steps; 75 policy updates (PPO/MAPPO/PI-MAPPO) vs 4,500 (value-based) (Table 7). Train each to convergence; report env steps; equalise gradient updates or justify.
- [ ] P1.8 Seeds ≥ 5 (prefer 10). Report mean + 95% bootstrap CI or IQM (Agarwal et al., NeurIPS 2021); significance test PI-MAPPO vs best baseline.
- [ ] P1.9 Evaluation scenarios must be stochastic and held-out: per-load random scaling (uniform scaling leaves LRR unchanged → std 0.000), different irradiance/wind days for train vs test (L615–616 admits one day for all), random initial SoC, random outage extent / black-start location; ≥ 100 evaluation episodes.
- [ ] P1.10 Report S1–S5 results (Table 4) per algorithm on 39- and 123-bus. None are reported; L424 mentions only S1–S3.
- [ ] P1.11 Full ablation: −features, −mask, −PBRS, −physics loss, −DAN (MLP concat), on ≥ 2 networks. Current ablation covers mask and PBRS only.
- [ ] P1.12 Add stressed cases where constraints bind, so VVR/FVR ≠ 0 for baselines (Fig. 5d–e: 0.0 for every algorithm, masked or not): long-line energisation (overvoltage), large load blocks / CLPU (frequency), low SoC (S4), low PV (S5).
- [ ] P1.13 Clarify train/validate/test across networks (L387–389; Fig. 1 labels). Zero-shot transfer is impossible with different action dimensions unless the architecture supports it. If retrained per network: call them three case studies and drop the generalisation claim (L498–499).
- [ ] P1.14 Reconcile "~1.8× / ~3.2× more episodes" (L500–502) with a fixed 300 episodes (Table 3).
- [ ] P1.15 "Dynamic agents" (contribution 4, Fig. 1 "Add new agent", §3.4): there are 5 fixed agents, one per action dimension. Either run an agent plug-in / IBR-trip experiment or remove the claim. Consider one agent per IBR unit or zone so the DAN's variable-neighbour design has a purpose.
- [ ] P1.16 Report the CS metric (defined §3.9, never reported). Report RoCoF (claimed L475, L496; no data shown).
- [ ] P1.17 Table 7: state which network; explain why PI-MAPPO (mask + DAN + physics loss) is faster than MAPPO (0.600 vs 0.827 s/episode); rewrite L547–549 ("~4% overhead per dimension, which affects LRR performance").
- [ ] P1.18 Fig. 6: LRR 56.2 → 37.3% when loads are added at fixed generation is an adequacy effect, not policy scalability. Report MW restored and priority-1 LRR.

## P2 — Formulation and theory

- [ ] P2.1 PBRS invariance (L328): with a fixed horizon T = 15 and non-zero terminal potential the shaping is not policy-invariant (Grześ, AAMAS 2017). Set Φ(terminal) = 0 or drop the claim. Cite Ng, Harada & Russell (ICML 1999); Devlin & Kudenko (AAMAS 2011) for the multi-agent case.
- [ ] P2.2 Φ (Eq. 34) = fraction of buses energised + fraction of load restored: contains no physics and duplicates Eq. 20. Make it physical (headroom, V/Q margin, SoC energy-to-go) or call it a progress potential.
- [ ] P2.3 Abstract L25–26: PBRS does not embed knowledge at "feature- and constraint-levels". Rewrite.
- [ ] P2.4 Physics loss (Eq. 35 / Eq. 41): define the auxiliary heads, their inputs and targets. L331–332 says predictions come "from Eq. (3)" and "Eq. (5)" — both are inequalities. Prefer equation residuals (mismatch of Eqs. 1–2, frequency model) so it is label-free; otherwise call it an auxiliary prediction loss.
- [ ] P2.5 Eq. 41 mixes actor, critic, entropy and physics terms over θ_i and φ in one loss; Alg. 1 l.24 "updates φ_v, φ_f, φ_p" (these are loss weights). Separate the losses; rename weights (β_v, β_f, β_p) and critic parameters (ψ). Resolve the φ / Φ / ϕ^PI collision.
- [ ] P2.6 Value-based masking (Alg. 2 l.9): `m ⊙ Q` fails for negative Q. Set invalid Q to −∞ in action selection AND in the bootstrap target.
- [ ] P2.7 Alg. 2 l.8 and l.18 put PPO and MAPPO in the ε-greedy / Q-update branch; only PI-MAPPO gets an on-policy update (l.25). Fix.
- [ ] P2.8 Alg. 1 l.11–15: `Φ(s_0)_now/prev` → Φ(s_{t+1}), Φ(s_t). L361–362: PI-MAPPO uses a rollout buffer, not a replay buffer.
- [ ] P2.9 Shared reward + cooperative agents = Dec-POMDP (Oliehoek & Amato 2016), not POSG (L238). Use one team reward r_t; drop the agent subscript in Eqs. 20–26 or define genuine per-agent rewards.
- [ ] P2.10 Define each agent's observation o_i. L261–262 and L267–268: if every agent sees s_t at execution it is not decentralised.
- [ ] P2.11 Action space: released code uses 11 power levels per unit → 16·11³·12 = 2.6×10⁵ on the 14-bus (matches "~10⁵"); the text's (n_pv+1) definition (§3.1, §4.1) gives 1.5×10³ / 5.2×10⁴ / 4.5×10⁶. Make §3.1, §4.1, L472–473, L490, L640 consistent.
- [ ] P2.12 Eqs. 13–14: summation index t clashes with conditioning time t. L245: γ ∈ [0, 1), not "γ^t : 0 → 1".
- [ ] P2.13 Eq. 18 uses C^j ("critical buses and loads", undefined) instead of the attention weight w^j of Eq. 19; add a value projection; define N_i for 5 fixed agents.
- [ ] P2.14 Eqs. 28, 32: "∀" used where "∨" is meant. Eqs. 29–31: malformed "0 ∀ ≤".
- [ ] P2.15 Eq. 36: state monotonicity as ∂Q_tot/∂Q_i ≥ 0 ∀ i.
- [ ] P2.16 Remove or evidence: "2.5× faster" (L355), "3–5× action-space shrink" (L357), "~4% overhead per dimension" (L548), "guaranteed improvement" (L351), "policy gradients never explode" (L473), "optimal reward shaping" (L477).
- [ ] P2.17 Publish all weights and hyper-parameters: w_r…w_d, w_n, w_l, β's, c_v, c_e, PPO epochs, minibatches, ε schedule, target update, grad clip, obs normalisation. Reconcile DAN sizes (L409–410: 128-128 and 32) with Table 3 (64–64).
- [ ] P2.18 Time-limit truncation in GAE: bootstrap at truncation (Pardo et al., ICML 2018). Released PPO sets next value to 0 at the time limit.

## P3 — Electrical model (what the environment must contain)

- [ ] P3.1 Make the GFM black-start unit the V-θ reference; delete the infinite bus. pandapower: `create_gen(..., slack=True, slack_weight ∝ 1/m_p)` + `runpp(distributed_slack=True)` reproduces steady-state droop sharing (verified, Appendix A).
- [ ] P3.2 After each power flow, check per-unit capability: √(P²+Q²) ≤ V·I_max·S_r; P ≤ P_avail(t); Q_min ≤ Q (under-excited limit for line charging); ESS power and SoC. Violation → mask via look-ahead PF, or terminate.
- [ ] P3.3 Frequency: a static PF gives no f, RoCoF or VSM response. Released code: f = 50 − 0.03·ΔP/ΣP_max·50; "RoCoF" = (f−50)/step_count; "SCR" = fraction of buses energised; ΔQ ≡ 0.0. Add an aggregated SFR/VSM model per switching event (Anderson & Mirheydar 1990): RoCoF₀ = ΔP_L·f₀/(2·H_eq·S_b); Δf_ss = ΔP_L/(Σ 1/R_k + D); nadir. State H_k (or M_k), D_k, m_p,k, n_q,k, LPF τ, and the COI aggregation. Note droop + LPF ⇔ VSM inertia ([12]).
- [ ] P3.4 State Δt per step (code: 1 min) and the hierarchy: primary = droop/VSM (ms–s); agent set-points = secondary; sequencing = minutes.
- [ ] P3.5 Put switches on the 5 transformers and traverse them in the energisation logic. Model transformer energisation as a rule (soft-start occupies k steps, or S_T ≤ α·S_GFM,online), cite [61]; a PF cannot show inrush.
- [ ] P3.6 Line energisation: once P3.1 is done, charging Q and Ferranti rise appear in the PF. Log Q_GFM and receiving-end V per step.
- [ ] P3.7 Cold-load pickup: P_L(t) = P_n·[1 + (k−1)·e^{−(t−t_r)/τ}]; state the load model (const-P / ZIP).
- [ ] P3.8 Maximum load step per action from RoCoF limit, nadir limit and headroom → electrical load mask (Eq. 32 is topological only).
- [ ] P3.9 Sync-check mask for closing between two live islands or closing a loop (ΔV, Δf, Δθ windows from the standard you adopt).
- [ ] P3.10 Protection = explicit failure conditions (UF/OF, UV/OV, RoCoF, overcurrent/thermal) → these define Eq. 25. Contribution (1) claims "protection relay logic"; §2 has none.
- [ ] P3.11 PV/WP as grid-forming needs headroom (deloading ρ) or DC-coupled storage. State the assumption; P_avail from irradiance/wind. S5 depends on it.
- [ ] P3.12 ESS: give MW and MVA ratings (Table 2 has MWh only). Eq. 7: e is SoC ∈ [0,1] but P·ΔT is MWh → divide by E_cap.
- [ ] P3.13 Eq. 12 limits aggregate PCC current. Add per-inverter limits (droop sharing can saturate one unit first). Fix units (√3·V·I in SI vs S = V·I in pu).
- [ ] P3.14 Eq. 11: inertia term and damping term are not dimensionally consistent as written; define M, D and units. L224: "oscillations coefficient" → damping coefficient.
- [ ] P3.15 Eqs. 1–2: with branch admittances and a line-status variable this is not the AC PF equation (no V_i²·g_ij term, no charging shunt). Use the Y_bus(x_t) injection form or the branch-flow form; add nodal balance P_i^G − x_i^load·P_i^L = P_i(V, θ, x).
- [ ] P3.16 L193 radiality Σx = N_b − N_s is a distribution (spanning-forest) condition and only necessary; IEEE-14/39 are meshed transmission. L599 calls all three "weakly meshed". Eq. 6 covers loads only — add monotonicity for buses and lines.
- [ ] P3.17 Document the modified test systems (appendix table: base kV/MVA, per-load MW/Mvar/priority, switchable branches, IBR bus and ratings, scaling, balanced-equivalent assumption). IEEE-39: 6,254.2 MW load vs 480 MW PV+WP yet LRR 63.2%. IEEE-123: 4.16 kV, 3.49 MW, unbalanced vs 525 MW of IBR; 91 load elements vs 95 loads in Table 2. case39 has 21 loads vs 28 in Table 2.
- [ ] P3.18 Define or delete "DER penetration (%)" (Table 2) — undefined in a blackout with no synchronous machines. Reconcile "single PCC" (L612–613) with 11 / 29 distributed units.
- [ ] P3.19 Features (Eq. 17): define ΔP, ΔQ as primary-response deviations (P_k − P_ref,k)/S_k — a converged PF has no bus mismatch. Add headroom, Q margin, SoC energy-to-go, RoCoF₀ estimate.
- [ ] P3.20 Objective vs reward: Eq. 8 cost terms are never defined and do not appear in Eq. 26 (Fig. 1 shows a third objective). SoC-zone penalties (L203–204) are not in Eq. 26. Add or delete. VVR (§3.9) cites Eq. 6; should be Eq. 3.
- [ ] P3.21 L229–230 is backwards. Current limiting does not protect differential relays; the issues are (i) a GFM cannot supply 6–10 pu inrush → soft energisation, (ii) ~1.1–1.2 pu fault current desensitises overcurrent protection.
- [ ] P3.22 Tie every limit to a cited standard / grid code: restoration V band, Δf_max (Fig. 5f shows ±0.5 Hz), RoCoF 1 Hz/s (L475), I_max 1.0–1.2 pu (L226), droop 3% / 4% (L221), thermal ratings.
- [ ] P3.23 L60–62: "voltage-controlled oscillator" → virtual oscillator control (VOC / dVOC).
- [ ] P3.24 L487: masks "filter out … voltage and frequency limit violations" — Eqs. 28–32 contain no V or f condition. Make true (P3.8, P3.9) or delete.

## P4 — Electrical results to add (new §4.x "Restoration sequence and electrical validation")

- [ ] P4.1 Single-line diagram per modified network: IBR buses, switchable branches, load priorities, energisation order of the best episode.
- [ ] P4.2 Step-wise time series for one representative episode (14- and 39-bus): P_PV / P_WP / P_ESS, restored MW by priority class, SoC, V_min / V_max, max line loading, I_GFM (pu) per unit, Q_GFM, f nadir and RoCoF per pickup.
- [ ] P4.3 Time-domain replay of the learned sequence: ANDES 2.0.0 ships REGF1 (droop), REGF2 (VSM), REGF3 (dVOC), REGCV1/2 (VSG) and IEEE-14/39 dynamic cases; or PowerFactory / Simulink RMS. Show f(t), V(t), I(t) for the largest load pickup and one line energisation.
- [ ] P4.4 Restoration metrics: ENS (MWh), time to restore priority-1 load, resilience-curve area ([16] is cited but unused), number of switching operations, curtailed energy, final SoC.
- [ ] P4.5 Table S1–S5 × algorithm with these electrical metrics.
- [ ] P4.6 Discuss Fig. 5f: PI-MAPPO has the largest |Δf| (0.19 Hz) of all algorithms.
- [ ] P4.7 Parameter table: m_p, n_q, H/M, D, I_max, τ, Δt, V band, Δf_max, RoCoF_max, efficiencies, SoC limits, reserve ρ, CLPU k and τ.

## P5 — Text and claims

- [ ] P5.1 Headline "60.63% over MAPPO": Table 6 shows DQN +60.1%, VDN +60.6%. Reframe around what is actually demonstrated.
- [ ] P5.2 L474 "highest number of energized buses across each system" contradicts Table 5 (39-bus: 19.0 vs VDN 20.7, QMIX 20.3, DQN 19.7; 123-bus: 53.0 vs IQL/QMIX 53.4).
- [ ] P5.3 L474–475 "0% VVR and FVR": true of every algorithm in Fig. 5d–e.
- [ ] P5.4 L476–479: value-decomposition performance attributed to "optimal reward shaping" and masking — state whether baselines have them (P1.2).
- [ ] P5.5 L531 "63.40%" vs Fig. 7b 0.614. L529 0.422 vs Table 5 0.420. Table 5 VDN 0.558 vs Table 6 VDN 0.559.
- [ ] P5.6 L458 "evaluation metrics in Table 6" → §3.9. L540–541 "As described in Section 4.1, the training time…" — not there.
- [ ] P5.7 L583–591 (key finding 4): the 14→123-bus drop used against VDN/DQN/PPO applies equally to PI-MAPPO (63.6 → 55.9%).
- [ ] P5.8 L74–77: "cannot be adopted" → scale poorly / need relaxations / must be re-solved. L88: IQL suffers from non-stationarity.
- [ ] P5.9 Table 1: "GFMI dynamics ●" and "Reproducible ●" only after P0 and P3 are done.
- [ ] P5.10 Remove the "Key strengths" box → separate Elsevier Highlights file (3–5 bullets, ≤ 85 characters). "et. al" → "et al.". Title uses minus signs for hyphens. ORCID empty.
- [ ] P5.11 Units: Δf in pu in text (L429: 0.0031 p.u.) vs Hz in Fig. 4d / 5f — pick one.
- [ ] P5.12 Language pass: L129, L241–243, L249, L507–509 and similar.

## P6 — Citations

- [ ] P6.1 L321: [57] (PINN state estimation) cited for invalid action masking → Huang & Ontañón (FLAIRS 2022).
- [ ] P6.2 L164: [54] (Raissi PINN) cited as a masking/shaping comparator.
- [ ] P6.3 L84–85: [22] (Gymnasium) cited as implementing DQN / DDQN / SAC — it is an environment API.
- [ ] P6.4 L85–86: [23] cited for "asynchronous information" → that is [30].
- [ ] P6.5 [24] for VDN / IQL → Sunehag et al. (AAMAS 2018); Tan (ICML 1993). [59] for MAPPO → Yu et al. (NeurIPS 2022). Add PPO (Schulman et al. 2017) and GAE (Schulman et al. 2016).
- [ ] P6.6 [58] for PBRS → add Ng 1999, Devlin & Kudenko 2011, Grześ 2017.
- [ ] P6.7 L231–232: "[51] recommends" MADRL agents as secondary controllers — [51] is a 2020 multi-ESS black-start control paper; check it says this.
- [ ] P6.8 Add: pandapower (if used), SFR model, a CLPU reference, grid-code / standard references for every limit.
- [ ] P6.9 Verify every 2025–2026 DOI resolves.

---

## Appendix A — GFM unit as reference with droop sharing (ran on pandapower 3.5.4)

```python
import numpy as np, pandapower as pp, pandapower.networks as pn

net = pn.case14()
net.gen.drop(net.gen.index, inplace=True)
net.ext_grid.drop(net.ext_grid.index, inplace=True)          # no infinite bus
units = {"GFM_ESS": (0, 90.0), "GFM_PV": (1, 55.0), "GFM_WP": (2, 70.0)}   # bus, S_rated MVA (illustrative)
for k, (bus, S) in units.items():
    pp.create_gen(net, bus=bus, p_mw=0.0, vm_pu=1.0, name=k, slack=True, slack_weight=S,
                  max_p_mw=S, min_p_mw=(-S if "ESS" in k else 0.0), sn_mva=S)
net.load["in_service"] = net.load.bus.isin([1, 2, 3, 4])      # partial-restoration snapshot
pp.runpp(net, distributed_slack=True)
```

Output:

```
converged: True | restored load = 171.3 MW
   name  p_mw  q_mvar  S_out  S_rated  I_pu_approx
GFM_ESS 72.57  -26.99  77.42     90.0         0.86
 GFM_PV 44.35   -3.95  44.52     55.0         0.81
 GFM_WP 56.44   23.69  61.21     70.0         0.87
steady-state droop df = -1.209 Hz for total pickup 173.4 MW (3% droop, P_ref = 0)
```

Reading: sharing follows ratings (droop), the ESS absorbs 27 Mvar of line charging, and −1.2 Hz is what primary droop alone leaves — the agent's set-point action is the secondary control that has to pull it back inside the band.
