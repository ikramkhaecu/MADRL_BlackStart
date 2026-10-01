"""Cooperative multi-agent black-start restoration environment (v2).

Five functional agents act once per step on a multi-discrete joint action
(manuscript Section 3.1):

    switch agent : close one switchable branch (line / transformer) or no-op
    PV agent     : dispatch level of the PV fleet            (secondary control)
    WP agent     : dispatch level of the wind fleet
    ESS agent    : charge / discharge level of the storage fleet
    load agent   : pick up (a block of) one de-energised load or no-op

Physics (all parameters in configs/default.yaml):
  * no infinite bus -- every island is formed by its GFM units; steady state
    from the droop-based AC power flow of ``sim.acpf`` (Eq. 9-10, 12);
  * frequency transient of every switching event from the aggregated VSM
    model of ``sim.frequency`` (Eq. 11): RoCoF(0+), nadir, settling value;
  * line charging / Ferranti rise, converter current limits, transformer
    inrush rule, cold-load pick-up, sync-check relay, ESS energy (Eq. 7);
  * protection: leaving the trip bands collapses the island (Eq. 25, r=-1).

Physics-informed components (each can be switched off for baselines and
ablations): feature vector phi^PI (Eq. 17), dynamic invalid-action mask with
admissible-step governor (Eq. 27-32), potential-based shaping (Eq. 33-34,
terminal potential zero).  The *reported* return is always the unshaped
environment return; the shaped reward is a separate training signal.
"""
from __future__ import annotations

import numpy as np

from .acpf import DroopPowerFlow, UnitState
from .frequency import FrequencyModel
from .networks import build

AGENTS = ("switch", "pv", "wp", "ess", "load")
KIND_OF = {"pv": "PV", "wp": "WP", "ess": "ESS"}
N_PHI = 8


class BlackStartEnv:
    def __init__(self, system="14", cfg=None, use_mask=True, use_features=True,
                 use_shaping=False, scenario=None, seed=0):
        from .config import load
        self.cfg = cfg or load()
        self.system = system
        self.grid = build(system, self.cfg)
        self.pf = DroopPowerFlow(self.grid)
        self.fm = FrequencyModel(self.cfg)
        self.use_mask, self.use_features, self.use_shaping = use_mask, use_features, use_shaping
        self.scenario = dict(scenario or {})
        c = self.cfg
        g = self.grid
        self.T = int(c["systems"][system].get("horizon", c["time"]["horizon"]))
        # mask limits: between soft and trip (Si et al. Table 1 insight — soft constraints are reward-guided)
        self.mask_v = (c["limits"].get("v_mask", [0.92, 1.08]))
        self.mask_f = c["limits"].get("df_mask_hz", 1.2)
        self.dt_min = float(c["systems"][system].get("dt_min", c["time"]["dt_min"]))
        self.dt_h = self.dt_min / 60.0
        self.base = g.base_mva
        self.sw_idx = np.flatnonzero(g.switchable)
        self.levels = dict(pv=np.array(c["dispatch"]["pv_levels"], float),
                           wp=np.array(c["dispatch"]["wp_levels"], float),
                           ess=np.array(c["dispatch"]["ess_levels"], float))
        self.unit_kind = np.array([u.kind for u in g.units])
        self.unit_bus = np.array([u.bus for u in g.units])
        self.unit_s = np.array([u.s_rated for u in g.units])
        self.unit_p = np.array([u.p_rated for u in g.units])
        self.unit_e = np.array([u.e_mwh for u in g.units])
        self.n_units = len(g.units)
        self._is_ess, self._is_pv = self.unit_kind == "ESS", self.unit_kind == "PV"
        self.n_act = [len(self.sw_idx) + 1, len(self.levels["pv"]) + 1,
                      len(self.levels["wp"]) + 1, len(self.levels["ess"]) + 1, g.n_load + 1]
        self.n_agents = 5
        self.w_prio = g.load_prio / g.load_prio.mean()
        self.rng = np.random.default_rng(seed)
        self.reset(seed)
        self.obs_dims = [len(o) for o in self._obs()]
        self.state_dim = len(self._state())
        self.phys_dim = g.n_bus + 1 + self.n_units      # aux-head targets (V, df, P_unit)

    # =============================================================== reset
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c, g, sc, rng = self.cfg, self.grid, self.scenario, self.rng
        s = c["scenario"]
        self.t = 0
        ls = rng.uniform(*s["load_scale"], size=g.n_load)
        self.load_p = g.load_p * ls
        self.load_q = g.load_q * ls
        n_str = int(sc.get("stress_loads", 0))
        if n_str:                                           # scalability stressor (v1 semantics)
            sel = np.random.default_rng(777).choice(g.n_load, min(n_str, g.n_load), replace=False)
            self.load_p[sel] *= sc.get("stress_scale", 1.5)
            self.load_q[sel] *= sc.get("stress_scale", 1.5)
        self.tot_w = float((self.w_prio * self.load_p).sum())
        self.load_x = np.zeros(g.n_load)
        self.clpu_excess = np.zeros(g.n_load)
        pv_rng = sc.get("pv_cf", s["pv_cf"]); wp_rng = sc.get("wp_cf", s["wp_cf"])
        self.cf = np.ones(self.n_units)
        for k, kd in enumerate(self.unit_kind):
            if kd == "PV": self.cf[k] = rng.uniform(*pv_rng)
            if kd == "WP": self.cf[k] = rng.uniform(*wp_rng)
        soc_rng = sc.get("soc_init", c["ess"]["soc_init"])
        if isinstance(soc_rng[0], (list, tuple)):           # union of intervals (S4)
            soc_rng = soc_rng[rng.integers(len(soc_rng))]
        self.soc = np.where(self.unit_kind == "ESS", rng.uniform(*soc_rng, size=self.n_units), 0.0)
        self.tripped = np.zeros(self.n_units, bool)
        self.trip_plan = None
        if sc.get("trip"):
            self.trip_plan = int(rng.integers(3, max(self.T - 3, 4)))
        self.level = dict(pv=0.0, wp=0.0, ess=0.0)
        mode = sc.get("blackstart", s["blackstart"])
        self.self_start = np.array([(kd == "ESS") or mode == "all" for kd in self.unit_kind])
        self.bus_on = np.zeros(g.n_bus, bool)
        self.bus_on[self.unit_bus[self.self_start]] = True
        self.br_closed = ~g.switchable.copy()
        self._propagate()
        self.online = self.bus_on[self.unit_bus] & ~self.tripped
        self.v_prev = np.full(g.n_bus, np.nan); self.th_prev = np.zeros(g.n_bus)
        self.ret = 0.0; self.ret_shaped = 0.0; self._w_restored = 0.0
        self.failed = False; self.fail_reason = ""
        self.n_switch_ops = 0; self.ens_mwh = 0.0; self.t_prio = None; self.shed_events = 0
        self.fillcfg = self.cfg.get("fill", dict(budget_frac_single=0.5, budget_frac_multi=0.3,
                                                 margin_f_hz=0.65, margin_v_pu=0.03,
                                                 max_rounds=40, retry_frac=0.5))
        self.log = dict(vviol=[], fviol=[], rocof=[], nadir=[], fss=[], vmin=[], vmax=[],
                        imax=[], qmargin=[], wasted=0, steps=0,
                        # -- operational trace (Sec. 5.5): admissible block, online base, islands
                        p_adm=[], s_eff=[], n_isl=[], picked=[], fsev=[])
        self.last_event = dict(rocof=0.0, nadir=0.0)
        if not hasattr(self.grid, '_orig_bs'):
            self.grid._orig_bs = self.grid.bs.copy()
        self._pref_plan = None
        self._solve()
        self._phi_prev = self._potential()
        self._masks = None
        return self._obs(), self._state()

    # ============================================================ topology
    def _propagate(self):
        g = self.grid
        changed = True
        while changed:
            changed = False
            for k in np.flatnonzero(self.br_closed):
                a, b = g.f[k], g.t[k]
                if self.bus_on[a] != self.bus_on[b]:
                    self.bus_on[a] = self.bus_on[b] = True
                    changed = True

    def _zone_if_closed(self, k):
        """Buses / branches that become live if switch k is closed from one live end."""
        g = self.grid
        a, b = g.f[k], g.t[k]
        start = b if self.bus_on[a] else a
        new_b, stack = {start}, [start]
        while stack:
            u = stack.pop()
            for kk in self._adj[u]:
                if self.br_closed[kk]:
                    v = g.t[kk] if g.f[kk] == u else g.f[kk]
                    if not self.bus_on[v] and v not in new_b:
                        new_b.add(v); stack.append(v)
        new_br = [k] + [kk for kk in np.flatnonzero(self.br_closed)
                        if (g.f[kk] in new_b or g.t[kk] in new_b)]
        return new_b, new_br

    @property
    def _adj(self):
        if not hasattr(self, "_adj_cache"):
            g = self.grid
            adj = [[] for _ in range(g.n_bus)]
            for k in range(g.n_br):
                adj[g.f[k]].append(k); adj[g.t[k]].append(k)
            self._adj_cache = adj
        return self._adj_cache

    # ========================================================= unit limits
    def _avail(self):
        """(p_min, p_max, p_ref) in MW for every unit from availability, SoC and levels."""
        c = self.cfg
        rho = c["gfm"]["reserve_frac"]
        pmin = np.zeros(self.n_units); pmax = np.zeros(self.n_units); pref = np.zeros(self.n_units)
        lo, hi = c["ess"]["soc_critical"]
        for k, kd in enumerate(self.unit_kind):
            if kd == "ESS":
                e = self.unit_e[k]
                dis = min(self.unit_p[k], max(self.soc[k] - 0.02, 0.0) * e * c["ess"]["eta_d"] / self.dt_h)
                chg = min(self.unit_p[k], max(0.98 - self.soc[k], 0.0) * e / c["ess"]["eta_c"] / self.dt_h)
                if self.soc[k] <= lo: dis = 0.0                    # pre-stop: only charge
                if self.soc[k] >= hi: chg = 0.0                    # pre-stop: only discharge
                pmax[k], pmin[k] = dis, -chg
                pref[k] = np.clip(self.level["ess"] * self.unit_p[k], -chg, dis)
            else:
                p_av = self.cf[k] * self.unit_p[k]
                pmax[k] = p_av
                pref[k] = self.level["pv" if kd == "PV" else "wp"] * (1 - rho) * p_av
        return pmin, pmax, pref

    # =================================================================== PF
    def _demand(self):
        return self.load_x * self.load_p + self.clpu_excess

    def _solve(self, need_sens=True):
        g, c = self.grid, self.cfg
        # Scale bus shunts by local load fraction (caps energise with load, not before)
        if hasattr(g, '_orig_bs'):
            for i in range(g.n_bus):
                g.bs[i] = g._orig_bs[i] * min(1.0, sum(
                    self.load_x[li] for li in range(g.n_load) if g.load_bus[li] == i) + 0.01)
            self.pf.ysh = (g.gs + 1j * g.bs) / g.base_mva
        dem = self._demand()
        pl = np.zeros(g.n_bus); ql = np.zeros(g.n_bus)
        np.add.at(pl, g.load_bus, dem / self.base)
        qf = np.divide(self.load_q, self.load_p, out=np.zeros_like(self.load_p), where=self.load_p > 0)
        np.add.at(ql, g.load_bus, dem * qf / self.base)
        pmin, pmax, pref = self._avail()
        self.online = self.bus_on[self.unit_bus] & ~self.tripped
        idx = np.flatnonzero(self.online)
        units = [UnitState(bus=int(self.unit_bus[k]), s=self.unit_s[k] / self.base,
                           p_ref=pref[k] / self.base, v0=c["gfm"]["v0_pu"], mp=c["gfm"]["droop_p"],
                           nq=c["gfm"]["droop_q"], p_min=pmin[k] / self.base, p_max=pmax[k] / self.base,
                           i_max=c["limits"]["i_cont_pu"]) for k in idx]
        live_br = self.br_closed & self.bus_on[g.f] & self.bus_on[g.t]
        r = self.pf.solve(self.bus_on, live_br, pl, ql, units, self.v_prev, self.th_prev, need_sens=need_sens)
        self.res, self.res_units = r, idx
        self._isl_units = None
        self.p_unit = np.zeros(self.n_units); self.q_unit = np.zeros(self.n_units)
        self.p_clamped = np.zeros(self.n_units, bool)
        self.p_unit[idx] = r.p_unit * self.base; self.q_unit[idx] = r.q_unit * self.base
        self.p_clamped[idx] = r.p_clamped
        self.pref, self.pmin, self.pmax = pref, pmin, pmax
        if r.ok:                                             # NaN marks buses without a solution yet
            self.v_prev = np.where(r.island >= 0, r.v, np.nan); self.th_prev = r.th
        return r

    def _island_of_bus(self, b):
        return int(self.res.island[b])

    def _s_eff(self, isl, exclude=None):
        """MVA of online units in island with active-power headroom (inertia + droop providers)."""
        m = self.online & (self.res.island[self.unit_bus] == isl) & ~self.p_clamped
        if exclude is not None:
            m = m.copy(); m[exclude] = False
        return float(self.unit_s[m].sum())

    def _headroom(self, isl):
        m = self.online & (self.res.island[self.unit_bus] == isl)
        return float(np.maximum(self.pmax[m] - self.p_unit[m], 0.0).sum())

    def _f_dev(self, isl_mask=None):
        """Steady-state frequency deviation (Hz) per bus."""
        return self.res.dw * self.fm.f0

    # ================================= droop balance with unit limits (predictor)
    def _kdroop(self):
        return self.unit_s / (self.cfg["gfm"]["droop_p"] * self.fm.f0)       # MW per Hz

    def _gen_at(self, df_hz, pref, m):
        return float(np.clip(pref[m] - self._kdroop()[m] * df_hz, self.pmin[m], self.pmax[m]).sum())

    def _f_pred(self, pref, need_mw, m):
        """Settling frequency deviation (Hz) of the units in mask m for a generation need
        (demand + losses).  sum_k clip(Pref_k - K_k df, Pmin_k, Pmax_k) is piecewise linear
        and decreasing in df, so it is solved exactly on its break-points."""
        if not m.any():
            return 0.0
        K, pr, lo, hi = self._kdroop()[m], pref[m], self.pmin[m], self.pmax[m]
        bp = np.sort(np.concatenate([(pr - hi) / K, (pr - lo) / K, [-5.0, 5.0]]))
        G = np.clip(pr[None, :] - K[None, :] * bp[:, None], lo[None, :], hi[None, :]).sum(1)
        if need_mw >= G[0]:
            return float(bp[0])
        if need_mw <= G[-1]:
            return float(bp[-1])
        j = int(np.searchsorted(-G, -need_mw))               # G is non-increasing
        g0, g1 = G[j - 1], G[j]
        if g0 == g1:
            return float(bp[j])
        return float(bp[j - 1] + (g0 - need_mw) / (g0 - g1) * (bp[j] - bp[j - 1]))

    def _pref_for(self, levels):
        """Set-points (MW) that a dict of fleet levels would produce right now."""
        ren = np.where(self._is_pv, levels["pv"], levels["wp"]) * (1 - self.cfg["gfm"]["reserve_frac"]) \
            * self.cf * self.unit_p
        return np.where(self._is_ess, np.clip(levels["ess"] * self.unit_p, self.pmin, self.pmax), ren)

    def _islands_units(self):
        if self._isl_units is None:
            isl = self.res.island[self.unit_bus]
            self._isl_units = [(i, self.online & (isl == i)) for i in np.unique(isl[self.online]) if i >= 0]
        return self._isl_units

    def _levels_ok(self, levels):
        """True if every island's predicted settling frequency stays inside 0.9 x band
        (or moves towards it) for the given fleet levels."""
        lim = 0.9 * self.mask_f  # dispatch mask limit
        pref = self._pref_for(levels)
        for i, m in self._islands_units():
            f_new = self._f_pred(pref, float(self.p_unit[m].sum()), m)
            f_now = float(self.res.dw[self.unit_bus[m]].mean() * self.fm.f0)
            if abs(f_new) > lim and abs(f_new) >= abs(f_now) - 1e-6:
                return False
        return True

    # ===================================== one-step look-ahead (shield) helpers
    _SNAP = ("br_closed", "bus_on", "load_x", "clpu_excess", "res", "res_units", "p_unit",
             "q_unit", "p_clamped", "pref", "pmin", "pmax", "v_prev", "th_prev", "online")

    def _snap(self):
        d = {k: getattr(self, k) for k in self._SNAP}
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}, dict(self.level)

    def _restore(self, snap):
        d, lev = snap
        for k, v in d.items():
            setattr(self, k, v)
        self.level = lev

    def _trial_ok(self, margin_v=0.001, margin_f=0.0):
        """Solve the present (tentative) configuration and check the *soft* limits."""
        L = self.cfg["limits"]
        r = self._solve(need_sens=False)
        if not r.ok:
            return False
        ev = r.island >= 0
        v = r.v[ev]
        if v.min() < self.mask_v[0] or v.max() > self.mask_v[1]:
            return False                                      # mask limit (between soft and trip)
        loading = np.maximum(np.abs(r.s_from), np.abs(r.s_to)) * self.base / np.maximum(self.grid.rate, 1e-6)
        if loading.max() > L["i_max_pu"]:                   # trip limit (thermal)
            return False
        return float(np.abs(r.dw[ev]).max() * self.fm.f0) <= 0.95 * (self.mask_f - margin_f)

    # =============================================== admissible load pick-up
    def _admissible_mw(self, li):
        """Largest nominal MW of load li that can be picked up now (0 if none)."""
        g, c, L = self.grid, self.cfg, self.cfg["limits"]
        b = g.load_bus[li]
        if not self.bus_on[b] or self.load_x[li] >= 1 - 1e-9 or not self.res.ok:
            return 0.0
        isl = self._island_of_bus(b)
        if isl < 0:
            return 0.0
        k_cl = c["load"]["clpu_k"] if c["load"]["clpu_enabled"] else 1.0
        remaining = (1 - self.load_x[li]) * self.load_p[li]
        df_pre = float(self.res.dw[b] * self.fm.f0)
        step = self.fm.max_step_mw(self._s_eff(isl), df_pre, L["rocof_soft_hz_s"], L["df_soft_hz"]) * 0.95
        # spinning-reserve rule (gamma_hat_min of Eq. 24): available MW of the island
        # must cover (1 + reserve) x demand *after* the pick-up incl. cold-load excess
        um = self.online & (r_isl := (self.res.island[self.unit_bus] == isl))
        cap = float(self.pmax[um].sum())
        gen = float(self.p_unit[um].sum())                   # present demand + losses of the island
        n_blk = max(int(self.dt_min * 60.0 / c["load"]["block_interval_s"]), 1)
        a = min(remaining, n_blk * step / k_cl, (cap / c["reward"]["adequacy_min"] - gen) / k_cl)
        if a <= 0:
            return 0.0
        sens = self.res.sens.get(isl)                       # voltage + settling frequency
        if sens is not None and sens["Jinv"] is not None:
            loc = int(np.flatnonzero(sens["buses"] == b)[0])
            qf = self.load_q[li] / self.load_p[li] if self.load_p[li] > 0 else 0.0
            dv, ddw = self.pf.predict_pickup(sens, loc, 1.0 / self.base, qf / self.base)  # per MW
            v_now = self.res.v[sens["buses"]]
            room_v = v_now - (self.mask_v[0] + 0.005)             # mask blocks at mask limit
            with np.errstate(divide="ignore", invalid="ignore"):
                lim = np.where(dv < -1e-12, room_v / (-dv), np.inf)
            a = min(a, float(max(lim.min(), 0.0)) / k_cl)
        pref = getattr(self, "_pref_plan", None)
        pref = self.pref if pref is None else pref
        room = self._gen_at(-0.9 * self.mask_f, pref, um) - gen * 1.01
        a = min(a, room / 1.03 / k_cl)                       # 3 % allowance for extra losses
        return max(a, 0.0)

    # =================================================================== masks
    def action_masks(self):
        """List of five boolean arrays (index 0 = no-op, always valid).  Cached per step."""
        if self._masks is not None:
            return self._masks
        g, c, L = self.grid, self.cfg, self.cfg["limits"]
        m = [np.zeros(n, bool) for n in self.n_act]
        for a in m:
            a[0] = True
        self.feas = [a.copy() for a in m]
        ok = self.res.ok and not self.failed
        # ---- switch agent (Eq. 28 + charging / voltage / inrush / sync-check)
        if ok:
            for j, k in enumerate(self.sw_idx):
                self.feas[0][j + 1] = self._switch_ok(k)
        # ---- dispatch agents (Eq. 29-31 + predicted settling frequency)
        for ai, ag in ((1, "pv"), (2, "wp"), (3, "ess")):
            sel = self.online & (self.unit_kind == KIND_OF[ag])
            if not ok or not sel.any():
                continue
            for j, lv in enumerate(self.levels[ag]):
                self.feas[ai][j + 1] = self._level_ok(ag, lv, sel)
        # ---- load agent (Eq. 32 + admissible step)
        self.adm = np.zeros(g.n_load)
        if ok:
            min_blk = c["load"]["min_pickup_frac"] * float(self.load_p.sum())
            for li in range(g.n_load):
                a = self._admissible_mw(li)
                rem = (1 - self.load_x[li]) * self.load_p[li]
                need = rem if c["load"]["pickup"] == "binary" else min(min_blk, rem)
                if a >= need - 1e-9 and rem > 1e-9:
                    self.adm[li] = a
                    self.feas[4][li + 1] = True
        if self.use_mask:
            self._masks = [f.copy() for f in self.feas]
        else:
            self._masks = [np.ones(n, bool) for n in self.n_act]
        return self._masks

    def _switch_ok(self, k):
        g, L, c = self.grid, self.cfg["limits"], self.cfg
        if self.br_closed[k]:
            return False
        a, b = g.f[k], g.t[k]
        la, lb = self.bus_on[a], self.bus_on[b]
        if not (la or lb):
            return False
        r = self.res
        if la and lb:                                       # loop closing / island merge: sync-check relay
            return self._sync_ok(a, b)
        live = a if la else b
        isl = self._island_of_bus(live)
        new_b, new_br = self._zone_if_closed(k)
        qc = float(g.b[new_br].sum()) * self.base * r.v[live] ** 2          # Mvar of charging
        um = self.online & (r.island[self.unit_bus] == isl)
        vk = r.v[self.unit_bus[um]]
        qcap = np.sqrt(np.maximum((L["i_cont_pu"] * self.unit_s[um] * vk) ** 2 - self.p_unit[um] ** 2, 0.0))
        margin = float((self.q_unit[um] + qcap).sum())      # Mvar that can still be absorbed
        if qc > 0.9 * margin:
            return False
        ferr = 1.0 + 0.5 * float(g.x[new_br].sum() * g.b[new_br].sum())     # cascaded-line bound
        ratio = (1.0 / g.tap[k]) if live == a else g.tap[k]                  # off-nominal tap of branch k
        ferr *= max(ratio, 1.0)
        sens = r.sens.get(isl)
        dv_live = 0.0
        if sens is not None and sens["Jinv"] is not None and qc > 0:
            loc = int(np.flatnonzero(sens["buses"] == live)[0])
            dv, _ = self.pf.predict_pickup(sens, loc, 0.0, -qc / self.base)
            dv_live = float(dv.max())
            if (r.v[sens["buses"]] + dv).max() > self.mask_v[1] - 0.005:
                return False
        if (r.v[live] + dv_live) * ferr > self.mask_v[1] - 0.005:
            return False
        st = float(g.trafo_sn[[kk for kk in new_br if g.is_trafo[kk]]].sum())
        if st > 0 and self._island_has_load(isl):
            s_on = float(self.unit_s[um].sum())
            if st * c["transformer"]["inrush_pu"] > L["i_max_pu"] * s_on:
                return False
        return True

    def _sync_ok(self, a, b):
        L, r = self.cfg["limits"], self.res
        if abs(r.v[a] - r.v[b]) > L["sync_dv_pu"]:
            return False
        if r.island[a] == r.island[b]:
            return abs(np.rad2deg(r.th[a] - r.th[b])) <= L["sync_dtheta_deg"]
        return abs((r.dw[a] - r.dw[b]) * self.fm.f0) <= L["sync_df_hz"]

    def _island_has_load(self, isl):
        g = self.grid
        return bool(((self.res.island[g.load_bus] == isl) & (self.load_x > 0)).any())

    def _level_ok(self, ag, lv, sel):
        c = self.cfg
        if ag == "ess":
            lo, hi = c["ess"]["soc_critical"]
            sc = self.soc[sel]
            if lv > 0 and (sc <= lo).all(): return False
            if lv < 0 and (sc >= hi).all(): return False
        lev = dict(self.level); lev[ag] = float(lv)
        return self._levels_ok(lev)

    # ==================================================================== step
    def step(self, actions):
        g, c, L = self.grid, self.cfg, self.cfg["limits"]
        if self._masks is None:
            self.action_masks()
        a_sw, a_pv, a_wp, a_ess, a_ld = [int(a) for a in actions]
        feas = self.feas
        wasted = 0
        events = {}                                          # island id -> MW step
        pre = self.res
        f_pre = pre.dw * self.fm.f0
        # ---- scenario S3: unit trip
        if self.trip_plan is not None and self.t == self.trip_plan:
            cand = [k for k in np.flatnonzero(self.online)
                    if (self.online & (pre.island[self.unit_bus] == pre.island[self.unit_bus[k]])).sum() > 1]
            if cand:
                k = int(self.rng.choice(cand))
                isl = int(pre.island[self.unit_bus[k]])
                events[isl] = events.get(isl, 0.0) + max(self.p_unit[k], 0.0)
                self.tripped[k] = True; self.online[k] = False
                self.trip_plan = None
            else:
                self.trip_plan += 1
        # ---- switch agent
        inrush_fail = False
        if a_sw > 0:
            k = self.sw_idx[a_sw - 1]
            aa, bb = g.f[k], g.t[k]
            if self.br_closed[k] or not (self.bus_on[aa] or self.bus_on[bb]):
                wasted += 1
            elif self.bus_on[aa] and self.bus_on[bb] and not self._sync_ok(aa, bb):
                wasted += 1                                   # sync-check relay blocks the closing
            else:
                if not (self.bus_on[aa] and self.bus_on[bb]):
                    live = aa if self.bus_on[aa] else bb
                    isl = self._island_of_bus(live)
                    _, new_br = self._zone_if_closed(k)
                    st = float(g.trafo_sn[[kk for kk in new_br if g.is_trafo[kk]]].sum())
                    um = self.online & (pre.island[self.unit_bus] == isl)
                    if st > 0 and self._island_has_load(isl) and \
                            st * c["transformer"]["inrush_pu"] > L["i_max_pu"] * float(self.unit_s[um].sum()):
                        inrush_fail = True
                snap = self._snap() if self.use_mask else None
                self.br_closed[k] = True
                self._propagate()
                if self.use_mask and (inrush_fail or not feas[0][a_sw] or not self._trial_ok()):
                    self._restore(snap)                       # shield: closing rejected (no-op)
                    inrush_fail = False
                else:
                    self.n_switch_ops += 1
        # ---- auto-close: (a) energise dead buses, (b) merge compatible islands
        if self.use_mask and not self.failed and self.t >= 2:  # skip t=1 to let initial state settle
            max_auto_mw = self.fm.max_step_mw(float(self.unit_s[self.online].sum()), 0.0,
                                              L["rocof_soft_hz_s"], L["df_soft_hz"]) * 0.3
            # Phase A: close live-to-dead switches (energise new buses)
            for _ in range(min(5, len(self.sw_idx))):
                closed_any = False
                for j2, k2 in enumerate(self.sw_idx):
                    if self.br_closed[k2]:
                        continue
                    a2, b2 = g.f[k2], g.t[k2]
                    if not (self.bus_on[a2] ^ self.bus_on[b2]):
                        continue
                    new_b, _ = self._zone_if_closed(k2)
                    new_load = sum(self.load_p[li] * (1 - self.load_x[li])
                                  for li in range(g.n_load)
                                  if g.load_bus[li] in new_b and not self.bus_on[g.load_bus[li]])
                    if new_load > max_auto_mw:
                        continue
                    snap_sw = self._snap()
                    self.br_closed[k2] = True
                    self._propagate()
                    if self._trial_ok(margin_v=self.fillcfg["margin_v_pu"], margin_f=self.fillcfg["margin_f_hz"]):  # limit = 0.5225 Hz (allows ~1% FVR)
                        self.n_switch_ops += 1
                        closed_any = True
                    else:
                        self._restore(snap_sw)
                if not closed_any:
                    break
            # Phase B: merge one pair of compatible islands per step
            n_isl = len(set(self.res.island[self.res.island >= 0].tolist()))
            if n_isl > 1:
                best_sw, best_score = -1, -1
                for j2, k2 in enumerate(self.sw_idx):
                    if self.br_closed[k2]:
                        continue
                    a2, b2 = g.f[k2], g.t[k2]
                    if not (self.bus_on[a2] and self.bus_on[b2]):
                        continue
                    ia, ib = self.res.island[a2], self.res.island[b2]
                    if ia == ib or ia < 0 or ib < 0:
                        continue
                    # frequency compatibility: both islands must be settled (df < 0.3 Hz)
                    da = abs(float(np.mean(self.res.dw[self.res.island == ia]) * self.fm.f0))
                    db = abs(float(np.mean(self.res.dw[self.res.island == ib]) * self.fm.f0))
                    if da > 0.3 or db > 0.3:
                        continue
                    # voltage compatibility
                    va = float(np.mean(self.res.v[self.res.island == ia]))
                    vb = float(np.mean(self.res.v[self.res.island == ib]))
                    if abs(va - vb) > 0.05:
                        continue
                    # score: prefer merging smaller islands first (less transient)
                    na = int(np.sum(self.res.island == ia))
                    nb = int(np.sum(self.res.island == ib))
                    score = min(na, nb)     # smaller merge = safer
                    if score > best_score:
                        best_sw, best_score = k2, score
                for _merge_i in range(2):     # up to 2 merges per step
                    if best_sw < 0:
                        break
                    snap_sw = self._snap()
                    self.br_closed[best_sw] = True
                    self._propagate()
                    if self._trial_ok(margin_v=self.fillcfg["margin_v_pu"], margin_f=self.fillcfg["margin_f_hz"]):  # limit = 0.5225 Hz (allows ~1% FVR)
                        self.n_switch_ops += 1
                        # find next best merge candidate after re-solving
                        self._solve(need_sens=True)
                        best_sw, best_score = -1, -1
                        for j2, k2 in enumerate(self.sw_idx):
                            if self.br_closed[k2]: continue
                            a2, b2 = g.f[k2], g.t[k2]
                            if not (self.bus_on[a2] and self.bus_on[b2]): continue
                            ia, ib = self.res.island[a2], self.res.island[b2]
                            if ia == ib or ia < 0 or ib < 0: continue
                            da = abs(float(np.mean(self.res.dw[self.res.island == ia]) * self.fm.f0))
                            db = abs(float(np.mean(self.res.dw[self.res.island == ib]) * self.fm.f0))
                            if da > 0.3 or db > 0.3: continue
                            va = float(np.mean(self.res.v[self.res.island == ia]))
                            vb = float(np.mean(self.res.v[self.res.island == ib]))
                            if abs(va - vb) > 0.05: continue
                            na = int(np.sum(self.res.island == ia))
                            nb = int(np.sum(self.res.island == ib))
                            score = min(na, nb)
                            if score > best_score:
                                best_sw, best_score = k2, score
                    else:
                        self._restore(snap_sw)
                        break
        # ---- dispatch agents.  With the safety layer on, changes are applied one by one
        #      and a change whose predicted settling frequency leaves the band is dropped
        #      (joint-action feasibility; v1 called this the "governor").
        plan = dict(self.level)
        for a, ag in ((a_pv, "pv"), (a_wp, "wp"), (a_ess, "ess")):
            if a > 0:
                if not (self.online & (self.unit_kind == KIND_OF[ag])).any():
                    wasted += 1
                    continue
                trial = dict(plan); trial[ag] = float(self.levels[ag][a - 1])
                if self.use_mask and not self._levels_ok(trial):
                    continue                                  # dropped by the safety layer
                plan = trial
        self.level = plan
        self._pref_plan = None
        # ---- operating point after switching / synchronising / re-dispatch and *before* the
        #      pick-up (sequence inside one step: switch -> units synchronise -> set-points ->
        #      load block).  The pick-up transient is evaluated against this point.
        mid = self._solve()
        mid_ok = mid.ok
        f_mid = mid.dw * self.fm.f0
        mid_clamped = self.p_clamped.copy(); mid_online = self.online.copy()
        self.p_unit_mid = self.p_unit.copy()
        # ---- load agent
        picked = 0.0
        ld_event = None
        if a_ld > 0:
            li = a_ld - 1
            b = g.load_bus[li]
            rem = (1 - self.load_x[li]) * self.load_p[li]
            if not self.bus_on[b] or rem <= 1e-9 or not mid_ok or mid.island[b] < 0:
                wasted += 1
            else:
                if self.use_mask:                             # admissible-step governor
                    amt = min(rem, self._admissible_mw(li)) if feas[4][a_ld] else 0.0   # under the planned set-points
                    if amt < min(c["load"]["min_pickup_frac"] * float(self.load_p.sum()), rem) - 1e-9:
                        amt = 0.0
                    if c["load"]["pickup"] == "binary" and amt < rem - 1e-9:
                        amt = 0.0
                else:
                    amt = rem
                k_cl = c["load"]["clpu_k"] if c["load"]["clpu_enabled"] else 1.0
                if self.use_mask and amt > 0:                 # look-ahead verification of the block
                    min_blk = min(c["load"]["min_pickup_frac"] * float(self.load_p.sum()), rem)
                    for _try in range(3):
                        snap = self._snap()
                        self.load_x[li] = min(1.0, self.load_x[li] + amt / self.load_p[li])
                        self.clpu_excess[li] += (k_cl - 1.0) * amt
                        good = self._trial_ok(margin_f=0.2)
                        self._restore(snap)
                        if good:
                            break
                        amt = amt / 2 if c["load"]["pickup"] != "binary" else 0.0
                        if amt < min_blk - 1e-9:
                            amt = 0.0
                            break
                    else:
                        amt = 0.0
                if amt <= 0:
                    wasted += 0 if self.use_mask else 1       # a shielded no-op is not an agent error
                else:
                    self.load_x[li] = min(1.0, self.load_x[li] + amt / self.load_p[li])
                    self.clpu_excess[li] += (k_cl - 1.0) * amt
                    ld_event = (int(mid.island[b]), k_cl * amt)
                    picked = amt
        _primary_picked = (picked > 0)

        # ---- multi-load greedy fill: add loads one at a time with PF re-solve
        if self.use_mask and not self.failed and mid_ok:
            k_cl = c["load"]["clpu_k"] if c["load"]["clpu_enabled"] else 1.0
            # SoC guard: if any ESS is nearly depleted, reduce fill budget
            _soc_ok = True
            if hasattr(self, 'soc') and self.soc is not None:
                for i in range(len(self.grid.units)):
                    if self.online[i] and self.grid.units[i].kind == 'ESS' and self.soc[i] < 0.15:
                        _soc_ok = False
                        break
            if not _soc_ok:
                _fill_budget = 0.0
            else:
                _fill_budget = self.fm.max_step_mw(float(self.unit_s[self.online].sum()), 0.0,
                                                   L["rocof_soft_hz_s"], L["df_soft_hz"])
            n_isl = len(set(self.res.island[self.res.island >= 0].tolist()))
            _fill_budget *= (self.fillcfg["budget_frac_single"] if n_isl <= 2
                             else self.fillcfg["budget_frac_multi"])
            _fill_budget -= picked
            _fill_tried = set()
            for _fill_round in range(min(int(self.fillcfg["max_rounds"]), g.n_load)):
                if _fill_budget <= 0.001 * float(self.load_p.sum()):
                    break
                order = np.argsort(-self.w_prio * (1 - self.load_x) * self.load_p)
                best_li, best_amt = -1, 0.0
                for li2 in order:
                    if li2 in _fill_tried:
                        continue
                    if (a_ld > 0 and li2 == a_ld - 1 and _primary_picked) or self.load_x[li2] >= 1 - 1e-9:
                        continue
                    b2 = g.load_bus[li2]
                    if not self.bus_on[b2] or mid.island[b2] < 0:
                        continue
                    rem2 = (1 - self.load_x[li2]) * self.load_p[li2]
                    amt2 = min(self._admissible_mw(li2), rem2, _fill_budget)
                    if amt2 >= 0.001 * float(self.load_p.sum()):
                        best_li, best_amt = li2, amt2
                        break                              # take the highest-priority admissible load
                if best_li < 0:
                    break
                snap2 = self._snap()
                self.load_x[best_li] = min(1.0, self.load_x[best_li] + best_amt / self.load_p[best_li])
                self.clpu_excess[best_li] += (k_cl - 1.0) * best_amt
                if self._trial_ok(margin_v=self.fillcfg["margin_v_pu"], margin_f=self.fillcfg["margin_f_hz"]):  # limit = 0.5225 Hz (allows ~1% FVR)
                    picked += best_amt
                    _fill_budget -= best_amt
                    if ld_event is None:
                        ld_event = (int(mid.island[g.load_bus[best_li]]), k_cl * best_amt)
                    else:
                        ld_event = (ld_event[0], ld_event[1] + k_cl * best_amt)
                    self._solve(need_sens=True)            # refresh for next load's admissible check
                else:
                    self._restore(snap2)
                    _fill_tried.add(best_li)
                    # try a smaller portion of this load
                    for _frac in (0.5,):       # one retry at 50%
                        smaller = best_amt * _frac
                        if smaller < 0.001 * float(self.load_p.sum()):
                            break
                        snap3 = self._snap()
                        self.load_x[best_li] = min(1.0, self.load_x[best_li] + smaller / self.load_p[best_li])
                        self.clpu_excess[best_li] += (k_cl - 1.0) * smaller
                        if self._trial_ok(margin_v=self.fillcfg["margin_v_pu"], margin_f=self.fillcfg["margin_f_hz"]):  # limit = 0.5225 Hz (allows ~1% FVR)
                            picked += smaller
                            _fill_budget -= smaller
                            if ld_event is None:
                                ld_event = (int(mid.island[g.load_bus[best_li]]), k_cl * smaller)
                            else:
                                ld_event = (ld_event[0], ld_event[1] + k_cl * smaller)
                            self._solve(need_sens=True)
                            break
                        else:
                            self._restore(snap3)
                    continue
        # ---- pre-check: if ld_event would cause a trip, reduce it
        if ld_event is not None and self.use_mask and not self.failed and mid_ok:
            isl_ld, dp_ld = ld_event
            um_ld = mid_online & (mid.island[self.unit_bus] == isl_ld)
            s_eff_ld = float(self.unit_s[um_ld & ~mid_clamped].sum())
            if s_eff_ld > 0:
                gen_ld = float(self.p_unit_mid[um_ld].sum())
                f_fin_ld = self._f_pred(self.pref, gen_ld + dp_ld * 1.02, um_ld)
                if abs(f_fin_ld) > 0.85 * L["df_trip_hz"]:
                    # shed fill loads until the predicted frequency is safe
                    safe_dp = 0.7 * L["df_trip_hz"] / abs(self.fm.settle_unit * self.fm.f0) * s_eff_ld / self.base if abs(self.fm.settle_unit) > 1e-9 else dp_ld
                    if safe_dp < dp_ld:
                        shed_mw = dp_ld - safe_dp
                        # undo the last-added fill loads
                        order_rev = np.argsort(self.w_prio * (self.load_x) * self.load_p)  # lowest priority first
                        for li_shed in order_rev:
                            if shed_mw <= 0:
                                break
                            shed_amt = min(self.load_x[li_shed] * self.load_p[li_shed], shed_mw)
                            if shed_amt > 0.1:
                                self.load_x[li_shed] = max(0, self.load_x[li_shed] - shed_amt / self.load_p[li_shed])
                                shed_mw -= shed_amt
                                picked -= shed_amt
                        ld_event = (isl_ld, max(0, dp_ld - (dp_ld - safe_dp)))
        # ---- frequency transient of this step's switching events (pre-event network)
        rocof_max = nadir_max = 0.0
        hard = None
        for isl, dp in events.items():
            s_eff = float(self.unit_s[self.online & (pre.island[self.unit_bus] == isl) & ~self.p_clamped].sum())
            ub = self.unit_bus[self.online & (pre.island[self.unit_bus] == isl)]
            fp = float(f_pre[ub].mean()) if len(ub) else 0.0
            ro, nad, _ = self.fm.event(dp, s_eff, fp)
            rocof_max = max(rocof_max, abs(ro)); nadir_max = max(nadir_max, abs(nad))
            if abs(ro) > L["rocof_trip_hz_s"] or abs(nad) > L["df_trip_hz"]:
                hard = "frequency_protection"
        if ld_event is not None:                              # pick-up against the mid point
            isl, dp = ld_event
            um = mid_online & (mid.island[self.unit_bus] == isl)
            s_eff = float(self.unit_s[um & ~mid_clamped].sum())
            fp = float(f_mid[self.unit_bus[um]].mean()) if um.any() else 0.0
            blk = dp
            if self.use_mask and s_eff > 0:
                # the governor energises the feeder in blocks (one every block_interval_s), each
                # inside the RoCoF limit; without the governor the whole load is one block
                blk = min(dp, 0.95 * self.fm.max_step_mw(s_eff, 0.0, L["rocof_soft_hz_s"], L["df_soft_hz"]))
            ro, _, _ = self.fm.event(blk, s_eff, 0.0)
            # nadir of the last block: second-order overshoot on top of the limit-aware droop balance
            gen_mid = float(self.p_unit_mid[um].sum())
            f_last = min(fp, self._f_pred(self.pref, gen_mid + (dp - blk) * 1.02, um))
            f_fin = self._f_pred(self.pref, gen_mid + dp * 1.02, um)
            nad = f_last + (f_fin - f_last) * (self.fm.nadir_unit / self.fm.settle_unit)
            rocof_max = max(rocof_max, abs(ro)); nadir_max = max(nadir_max, abs(nad))
            if abs(ro) > L["rocof_trip_hz_s"] or abs(nad) > L["df_trip_hz"]:
                hard = "frequency_protection"
        if inrush_fail:
            hard = "inrush_overcurrent"
        if not mid_ok and hard is None:
            hard = mid.reason or "pf_not_converged"
        # ---- new steady state
        r = self._solve()
        info = self._evaluate(r, rocof_max, nadir_max, hard, wasted, picked)
        return self._obs(), self._state(), info["reward"], info["done"], info

    # ---------------------------------------------------------------- evaluate
    def _evaluate(self, r, rocof_max, nadir_max, hard, wasted, picked):
        g, c, L, R = self.grid, self.cfg, self.cfg["limits"], self.cfg["reward"]
        w_before = getattr(self, "_w_restored", 0.0)
        if not r.ok and hard is None:
            hard = r.reason
        vviol = fviol = lviol = 0.0
        vmin = vmax = 1.0; imax = 0.0; qmarg = 1.0; fss = 0.0
        r_pi = 0.0
        if r.ok:
            ev = r.island >= 0
            v = r.v[ev]
            vmin, vmax = float(v.min()), float(v.max())
            over = np.maximum(np.maximum(v - L["v_soft"][1], L["v_soft"][0] - v), 0.0)
            vviol = float((over > 0).mean())
            if vmin < L["v_trip"][0] or vmax > L["v_trip"][1]:
                hard = hard or "voltage_protection"
            fss = float(np.abs(r.dw[ev]).max() * self.fm.f0)
            if fss > L["df_trip_hz"]:
                hard = hard or "frequency_protection"
            f_ex = max(max(fss, nadir_max) - L["df_soft_hz"], 0.0) / L["df_soft_hz"] \
                + max(rocof_max - L["rocof_soft_hz_s"], 0.0) / L["rocof_soft_hz_s"]
            fviol = 1.0 if f_ex > 0 else 0.0
            live = np.abs(r.s_from) > 0
            load_pu = np.maximum(np.abs(r.s_from), np.abs(r.s_to)) * self.base / np.maximum(g.rate, 1e-6)
            lviol = float(np.maximum(load_pu[live] - L["line_loading_soft"], 0.0).mean()) if live.any() else 0.0
            on = self.online
            vk = r.v[self.unit_bus[on]]
            s_out = np.hypot(self.p_unit[on], self.q_unit[on]) / np.maximum(self.unit_s[on] * vk, 1e-9)
            imax = float(s_out.max()) if on.any() else 0.0
            qcap = np.sqrt(np.maximum((L["i_cont_pu"] * self.unit_s[on] * vk) ** 2 - self.p_unit[on] ** 2, 0.0))
            qmarg = float(((qcap - np.abs(self.q_unit[on])) / np.maximum(self.unit_s[on], 1e-9)).min()) if on.any() else 0.0
            dP = float(np.abs(self.p_unit[on] - self.pref[on]).sum() / max(self.unit_s[on].sum(), 1e-9))
            dQ = float(np.maximum(np.abs(self.q_unit[on]) - 0.8 * qcap, 0.0).sum() / max(self.unit_s[on].sum(), 1e-9))
            dem = float(self._demand().sum())
            gam = float(self.pmax[on].sum() / dem) if dem > 1e-9 else R["adequacy_min"]
            r_pi = -(dP + dQ + max(R["adequacy_min"] - gam, 0.0))
            r_v = -float((over / 0.05).mean())
            r_f = -f_ex
        else:
            r_v = r_f = 0.0
        # ---- state of charge (Eq. 7) and zones
        r_soc = 0.0
        ess = np.flatnonzero(self.unit_kind == "ESS")
        if r.ok:
            for k in ess:
                p = self.p_unit[k]
                de = (p / c["ess"]["eta_d"] if p >= 0 else p * c["ess"]["eta_c"]) * self.dt_h
                self.soc[k] = float(np.clip(self.soc[k] - de / self.unit_e[k], 0.0, 1.0))
        n_lo, n_hi = c["ess"]["soc_normal"]; c_lo, c_hi = c["ess"]["soc_critical"]
        zone = np.where((self.soc[ess] < c_lo) | (self.soc[ess] > c_hi), 1.0,
                        np.where((self.soc[ess] < n_lo) | (self.soc[ess] > n_hi), 0.5, 0.0))
        socv = float((zone >= 1.0).mean()) if len(ess) else 0.0
        r_soc = -float(zone.mean()) if len(ess) else 0.0
        # ---- recovery (Eq. 20), collapse loses everything
        w_now = float((self.w_prio * self.load_x * self.load_p).sum() / self.tot_w)
        self.t += 1
        done = False
        r_done = 0.0
        if hard is not None:
            self.failed, self.fail_reason = True, hard
            w_now = 0.0
            done, r_done = True, -1.0
        elif self.load_x.min() >= 1 - 1e-9 or self.t >= self.T:
            done = True
            crit = self.grid.load_prio >= max(self.cfg["load"]["priority_levels"])
            clean = (sum(self.log["vviol"]) + vviol + sum(self.log["fviol"]) + fviol) == 0
            r_done = 1.0 if (clean and self.load_x[crit].min() >= 1 - 1e-9) else 0.0
        r_rec = w_now - w_before
        self._w_restored = w_now
        ps = self.cfg["time"]["horizon"] / self.T            # per-step penalties are horizon-normalised
        self.last_terms = dict(rec=R["w_rec"] * r_rec, v=ps * R["w_v"] * r_v, f=ps * R["w_f"] * r_f,
                               line=-ps * R["w_line"] * lviol, pi=ps * R["w_pi"] * r_pi,
                               soc=ps * R["w_soc"] * r_soc, done=R["w_done"] * r_done,
                               inf=-ps * R["w_inf"] * wasted / self.n_agents)
        reward = float(sum(self.last_terms.values()))
        # ---- potential-based shaping (Eq. 33-34), terminal potential = 0
        phi = 0.0 if done else self._potential()
        shaped = reward + c["pbrs"]["gamma"] * phi - self._phi_prev
        self._phi_prev = phi
        self.ret += reward; self.ret_shaped += shaped
        # ---- book-keeping
        lg = self.log
        lg["vviol"].append(vviol); lg["fviol"].append(fviol); lg["rocof"].append(rocof_max)
        lg["nadir"].append(nadir_max); lg["fss"].append(fss); lg["vmin"].append(vmin)
        lg["vmax"].append(vmax); lg["imax"].append(imax); lg["qmargin"].append(qmarg)
        lg["wasted"] += wasted; lg["steps"] += 1
        # -- Delta P^adm on the online-unsaturated MVA base, and the frequency-severity index
        _seff = float(self.unit_s[self.online & ~self.p_clamped].sum())
        _padm = self.fm.max_step_mw(_seff, 0.0, self.cfg["limits"]["rocof_soft_hz_s"],
                                    self.cfg["limits"]["df_soft_hz"]) if _seff > 0 else 0.0
        lg["s_eff"].append(_seff); lg["p_adm"].append(_padm)
        lg["n_isl"].append(int(len({int(i) for i in self.res.island if i >= 0})))
        lg["picked"].append(float(picked))
        _dfl = self.cfg["limits"]["df_soft_hz"]; _rl = self.cfg["limits"]["rocof_soft_hz_s"]
        lg["fsev"].append(max(0.0, max(nadir_max, fss) / _dfl - 1.0,
                              rocof_max / _rl - 1.0))
        served = 0.0 if self.failed else float((self.load_x * self.load_p).sum())
        self.ens_mwh += (float(self.load_p.sum()) - served) * self.dt_h
        crit = self.grid.load_prio >= max(self.cfg["load"]["priority_levels"])
        if self.t_prio is None and not self.failed and self.load_x[crit].min() >= 1 - 1e-9:
            self.t_prio = self.t
        # ---- decay / random walks for the next step
        if c["load"]["clpu_enabled"]:
            self.clpu_excess *= np.exp(-self.dt_min / c["load"]["clpu_tau_min"])
        walk = self.rng.normal(0.0, c["scenario"]["cf_walk_std"] * np.sqrt(self.dt_min / c["time"]["dt_min"]), self.n_units)
        ren = self.unit_kind != "ESS"
        self.cf[ren] = np.clip(self.cf[ren] + walk[ren], 0.05, 1.0)
        self.last_event = dict(rocof=rocof_max, nadir=nadir_max)
        if not done:
            self._solve_after_walk()
        self._masks = None
        self._pref_plan = None
        info = dict(reward=reward, shaped=shaped, done=done, lrr=self.lrr(), lrr_w=w_now,
                    buses=int(self.bus_on.sum()) if not self.failed else 0, vvio=vviol, fvio=fviol,
                    soc_v=socv, rocof=rocof_max, nadir=nadir_max, fss=fss, vmin=vmin, vmax=vmax,
                    imax=imax, qmargin=qmarg, wasted=wasted, picked_mw=picked, failed=self.failed,
                    fail_reason=self.fail_reason, phys=self._phys_target(),
                    droop=self.droop_terms())
        return info

    def _solve_after_walk(self):
        """Availability moved -> limits moved; refresh the operating point (no event).
        If the renewables dropped below the demand, under-frequency load shedding
        removes the lowest-priority restored load instead of collapsing the island."""
        r = self._solve()
        tries = 0
        while not r.ok and tries < 12 and (self.load_x > 0).any():
            dem = self._demand()
            need = max(0.10 * dem.sum(), dem.sum() * 1.03 - float(self.pmax[self.online].sum()))
            order = np.lexsort((-(self.load_x * self.load_p), self.grid.load_prio))    # lowest priority first
            for li in order:
                if need <= 0:
                    break
                if self.load_x[li] > 0:
                    need -= dem[li]
                    self.load_x[li] = 0.0; self.clpu_excess[li] = 0.0
            self.shed_events += 1
            tries += 1
            r = self._solve()
        if not r.ok and not self.failed:
            self.failed, self.fail_reason = True, r.reason

    # ============================================================== potentials
    def _potential(self):
        p = self.cfg["pbrs"]
        w = float((self.w_prio * self.load_x * self.load_p).sum() / self.tot_w)
        return p["w_l"] * w + p["w_n"] * float(self.bus_on.mean())

    def lrr(self):
        return 0.0 if self.failed else float((self.load_x * self.load_p).sum() / self.load_p.sum())

    # ============================================================ observations
    def _phi(self):
        """Physics-informed feature vector phi^PI (Eq. 17, extended)."""
        if not self.use_features or not self.res.ok:
            return np.zeros(N_PHI, np.float32)
        L, on, r = self.cfg["limits"], self.online, self.res
        s_on = max(self.unit_s[on].sum(), 1e-9)
        dP = float((self.p_unit[on] - self.pref[on]).sum() / s_on)
        vk = r.v[self.unit_bus[on]]
        qcap = np.sqrt(np.maximum((L["i_cont_pu"] * self.unit_s[on] * vk) ** 2 - self.p_unit[on] ** 2, 0.0))
        dQ = float(self.q_unit[on].sum() / max(qcap.sum(), 1e-9))
        ev = r.island >= 0
        dV = float(r.v[ev].mean() - 1.0) * 10.0
        dem = float(self._demand().sum())
        gam = min(float(self.pmax[on].sum() / dem), 5.0) / 5.0 if dem > 1e-9 else 1.0
        step = self.fm.max_step_mw(float(self.unit_s[on & ~self.p_clamped].sum()),
                                   float(-np.abs(r.dw[ev]).max() * self.fm.f0),
                                   L["rocof_soft_hz_s"], L["df_soft_hz"]) / max(self.load_p.sum(), 1e-9)
        head = float(np.maximum(self.pmax[on] - self.p_unit[on], 0).sum() / max(self.load_p.sum(), 1e-9))
        ess = on & (self.unit_kind == "ESS")
        e2go = float((self.soc[ess] * self.unit_e[ess]).sum() / max(self.unit_e[self.unit_kind == "ESS"].sum(), 1e-9))
        return np.array([dP, dQ, self.last_event["rocof"] / L["rocof_soft_hz_s"], dV, gam,
                         step * 10.0, head, e2go], np.float32)

    def _sys_block(self):
        r, L = self.res, self.cfg["limits"]
        ev = r.island >= 0
        f = float(np.abs(r.dw[ev]).max() * self.fm.f0 * np.sign(r.dw[ev][np.abs(r.dw[ev]).argmax()])) if (r.ok and ev.any()) else 0.0
        return np.concatenate([[f / L["df_soft_hz"], self.t / self.T, float(self.bus_on.mean()),
                                self.lrr(), self.level["pv"], self.level["wp"], self.level["ess"]],
                               self._phi()]).astype(np.float32)

    def _obs(self):
        g, r = self.grid, self.res
        if self._masks is None or not hasattr(self, "feas"):
            try:
                self.action_masks()
            except Exception:
                self.feas = [np.zeros(n, bool) for n in self.n_act]; self.adm = np.zeros(g.n_load)
        sysb = self._sys_block()
        v = np.where(r.island >= 0, np.nan_to_num(r.v), 0.0)
        sw = self.sw_idx
        one = (self.bus_on[g.f[sw]] ^ self.bus_on[g.t[sw]]).astype(float)
        o_sw = np.concatenate([sysb, self.br_closed[sw].astype(float), one,
                               self.feas[0][1:].astype(float), g.is_trafo[sw].astype(float),
                               self.bus_on.astype(float), (v - 1.0) * 10.0])
        outs = [o_sw]
        for ag in ("pv", "wp", "ess"):
            sel = self.unit_kind == KIND_OF[ag]
            s = np.maximum(self.unit_s[sel], 1e-9)
            blk = [self.online[sel].astype(float), self.p_unit[sel] / s, self.q_unit[sel] / s,
                   self.pref[sel] / s, self.pmax[sel] / s, self.tripped[sel].astype(float),
                   (v[self.unit_bus[sel]] - 1.0) * 10.0]
            if ag == "ess":
                blk += [self.soc[sel], self.pmin[sel] / s]
            ai = AGENTS.index(ag)
            outs.append(np.concatenate([sysb] + blk + [self.feas[ai][1:].astype(float)]))
        tot = max(self.load_p.sum(), 1e-9)
        o_ld = np.concatenate([sysb, self.load_x, self.bus_on[g.load_bus].astype(float),
                               g.load_prio / g.load_prio.max(),
                               (1 - self.load_x) * self.load_p / tot * 10.0,
                               self.feas[4][1:].astype(float), self.adm / tot * 10.0])
        outs.append(o_ld)
        return [o.astype(np.float32) for o in outs]

    def _state(self):
        g, r = self.grid, self.res
        v = np.where(r.island >= 0, np.nan_to_num(r.v), 0.0)
        s = np.maximum(self.unit_s, 1e-9)
        return np.concatenate([self._sys_block(), self.bus_on.astype(float), (v - 1.0) * 10.0 * self.bus_on,
                               self.br_closed[self.sw_idx].astype(float), self.load_x,
                               self.online.astype(float), self.p_unit / s, self.q_unit / s,
                               self.pref / s, self.pmax / s, self.soc]).astype(np.float32)

    def _phys_target(self):
        """Targets of the physics heads of the critic: V (all buses), signed settling
        frequency deviation (in units of the soft band), P_unit / S_unit."""
        r = self.res
        v = np.where(r.island >= 0, np.nan_to_num(r.v), 0.0)
        f = 0.0
        if r.ok and (r.island >= 0).any():
            dw = r.dw[r.island >= 0]
            f = float(dw[np.abs(dw).argmax()] * self.fm.f0 / self.cfg["limits"]["df_soft_hz"])
        return np.concatenate([v, [f], self.p_unit / np.maximum(self.unit_s, 1e-9)]).astype(np.float32)

    def droop_terms(self):
        """(P_ref/S, K=1/(m_p) scaled, free-mask) for the droop-law residual of the critic."""
        s = np.maximum(self.unit_s, 1e-9)
        free = (self.online & ~self.p_clamped).astype(np.float32)
        return (self.pref / s).astype(np.float32), free

    # ================================================================ summary
    def summary(self):
        lg = self.log
        n = max(lg["steps"], 1)
        return dict(ret=self.ret, ret_shaped=self.ret_shaped, lrr=self.lrr(),
                    lrr_w=getattr(self, "_w_restored", 0.0),
                    mw=0.0 if self.failed else float((self.load_x * self.load_p).sum()),
                    buses=0 if self.failed else int(self.bus_on.sum()),
                    vvr=float(np.mean(lg["vviol"])) if lg["vviol"] else 0.0,
                    fvr=float(np.mean(lg["fviol"])) if lg["fviol"] else 0.0,
                    rocof=float(np.max(lg["rocof"])) if lg["rocof"] else 0.0,
                    nadir=float(np.max(lg["nadir"])) if lg["nadir"] else 0.0,
                    fss=float(np.max(lg["fss"])) if lg["fss"] else 0.0,
                    vmin=float(np.min(lg["vmin"])) if lg["vmin"] else 1.0,
                    vmax=float(np.max(lg["vmax"])) if lg["vmax"] else 1.0,
                    imax=float(np.max(lg["imax"])) if lg["imax"] else 0.0,
                    qmargin=float(np.min(lg["qmargin"])) if lg["qmargin"] else 1.0,
                    wasted=lg["wasted"] / n, shed=self.shed_events, switch_ops=self.n_switch_ops, ens_mwh=self.ens_mwh,
                    t_prio=(self.t_prio if self.t_prio is not None else self.T + 1),
                    soc_end=float(self.soc[self.unit_kind == "ESS"].mean()),
                    fvr_sev=float(np.mean(lg["fsev"])) if lg["fsev"] else 0.0,
                    cap_bind=float(np.mean([pk >= 0.95 * pa for pk, pa in
                                            zip(lg["picked"], lg["p_adm"]) if pa > 1e-6]))
                              if any(pa > 1e-6 for pa in lg["p_adm"]) else 0.0,
                    s_eff_end=float(lg["s_eff"][-1]) if lg["s_eff"] else 0.0,
                    p_adm_end=float(lg["p_adm"][-1]) if lg["p_adm"] else 0.0,
                    n_isl_end=int(lg["n_isl"][-1]) if lg["n_isl"] else 0,
                    failed=float(self.failed), fail_reason=self.fail_reason,
                    cs=float((not self.failed) and sum(lg["vviol"]) == 0 and sum(lg["fviol"]) == 0),
                    steps=lg["steps"])
