"""Test-system construction (IEEE 14 / 39 / 118) as per-unit :class:`GridData`.

The three systems are *modified* IEEE cases; every modification is driven by
``configs/default.yaml`` (with a provenance tag) and summarised by
:func:`describe`, which is what Table 2 / the appendix of the manuscript must
report.  Nothing here touches the series impedances or topology of the
original data.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger("pandapower").setLevel(logging.ERROR)


@dataclass
class Unit:
    kind: str            # "ESS" | "PV" | "WP"
    bus: int
    p_rated: float       # MW   (ESS: max charge/discharge power)
    s_rated: float       # MVA
    e_mwh: float = 0.0   # ESS energy capacity


@dataclass
class GridData:
    name: str
    base_mva: float
    bus_names: list
    vn_kv: np.ndarray
    f: np.ndarray; t: np.ndarray
    r: np.ndarray; x: np.ndarray; b: np.ndarray
    tap: np.ndarray; shift: np.ndarray
    rate: np.ndarray                 # MVA thermal rating per branch
    is_trafo: np.ndarray
    trafo_sn: np.ndarray             # MVA (0 for lines)
    switchable: np.ndarray           # bool per branch
    br_names: list
    gs: np.ndarray; bs: np.ndarray   # bus shunts, MW / Mvar at 1 pu
    load_bus: np.ndarray; load_p: np.ndarray; load_q: np.ndarray
    load_prio: np.ndarray
    units: list = field(default_factory=list)
    coords: np.ndarray | None = None
    notes: dict = field(default_factory=dict)

    @property
    def n_bus(self): return len(self.bus_names)
    @property
    def n_br(self): return len(self.f)
    @property
    def n_load(self): return len(self.load_bus)
    @property
    def total_p(self): return float(self.load_p.sum())


# --------------------------------------------------------------- pandapower
def _from_pandapower(case: str):
    import pandapower as pp
    import pandapower.networks as pn
    from pandapower.converter.pypower.to_ppc import to_ppc
    net = getattr(pn, case)()
    ppc = to_ppc(net, init="flat", calculate_voltage_angles=True)
    br, bus = ppc["branch"].real, ppc["bus"].real
    nl = len(net.line)
    pp.runpp(net)                                    # base case -> surrogate ratings
    flow = np.concatenate([
        np.hypot(net.res_line.p_from_mw, net.res_line.q_from_mvar).to_numpy(),
        np.hypot(net.res_trafo.p_hv_mw, net.res_trafo.q_hv_mvar).to_numpy()])
    tap = br[:, 8].copy(); tap[tap == 0] = 1.0
    return dict(
        name=case, base_mva=float(ppc["baseMVA"]), vn_kv=bus[:, 9].copy(),
        bus_names=[str(int(i)) for i in bus[:, 0]],
        f=br[:, 0].astype(int), t=br[:, 1].astype(int), r=br[:, 2].copy(),
        x=br[:, 3].copy(), b=br[:, 4].copy(), tap=tap, shift=np.deg2rad(br[:, 9]),
        rate=br[:, 5].copy(), base_flow=flow,
        is_trafo=np.arange(len(br)) >= nl,
        br_names=[f"L{i}" for i in range(nl)] + [f"T{i}" for i in range(len(br) - nl)],
        load_bus=net.load.bus.to_numpy().astype(int),
        load_p=net.load.p_mw.to_numpy(float), load_q=net.load.q_mvar.to_numpy(float),
        gs=bus[:, 4].copy(), bs=bus[:, 5].copy(), coords=None,
        gen_bus=np.concatenate([net.ext_grid.bus.to_numpy(), net.gen.bus.to_numpy()]).astype(int),
        gen_cap=np.concatenate([np.full(len(net.ext_grid), np.inf), net.gen.max_p_mw.to_numpy(float)]),
        shunt_list=[(int(r.bus), float(r.q_mvar)) for r in net.shunt.itertuples()],
    )


# -------------------------------------------------------------------- build
def build(system: str, cfg: dict) -> GridData:
    sc, g = cfg["systems"][system], cfg["gfm"]
    d = _from_pandapower(sc["case"])
    n, m = len(d["bus_names"]), len(d["f"])
    notes = {}

    # ---- loads: scale, split, priorities
    lb, lp, lq = d["load_bus"].copy(), d["load_p"] * sc["load_scale_base"], d["load_q"] * sc["load_scale_base"]
    k_split = int(sc.get("split_largest_loads", 0))
    if k_split:
        big = np.argsort(-lp)[:k_split]
        lp[big] /= 2; lq[big] /= 2
        lb, lp, lq = np.append(lb, lb[big]), np.append(lp, lp[big]), np.append(lq, lq[big])
        notes["split_loads"] = [int(i) for i in big]
    rng = np.random.RandomState(cfg["load"]["priority_seed"])
    prio = rng.choice(cfg["load"]["priority_levels"], size=len(lp))

    # ---- branches: shunt compensation, ratings
    b = d["b"] * (1.0 - sc.get("line_shunt_comp", 0.0))
    rate = d["rate"].astype(float).copy()
    placeholder = rate > 5000.0                                   # MATPOWER "9900"
    if placeholder.any():
        rc = cfg["ratings"]
        rate[placeholder] = np.maximum(rc["surrogate_factor"] * d["base_flow"][placeholder],
                                       rc["surrogate_min_mva"])
        notes["surrogate_ratings"] = int(placeholder.sum())
    trafo_sn = np.where(d["is_trafo"], rate, 0.0)

    # ---- units
    units = []
    def mk(kind, bus, p=None, e=None):
        if kind == "ESS":
            p = e / cfg["ess"]["duration_h"]
        units.append(Unit(kind, int(bus), float(p), float(p) / g["pf_rated"], float(e or 0.0)))
    if "units" in sc:
        for u in sc["units"]:
            mk(u["type"], u["bus"], u.get("p_mw"), u.get("e_mwh"))
    else:                                # placement rule: generator buses by capability (slack first)
        cnt = sc["unit_counts"]; total = sum(cnt.values())
        order = np.argsort(-d["gen_cap"], kind="stable")
        buses = [int(b) for b in d["gen_bus"][order][:total]]
        kinds, pool = [], dict(cnt)
        while len(kinds) < total:                                  # ESS, PV, WP cyclic
            for kd in ("ESS", "PV", "WP"):
                if pool.get(kd, 0) > 0 and len(kinds) < total:
                    kinds.append(kd); pool[kd] -= 1
        for kd, bu in zip(kinds, buses):
            mk(kd, bu, sc["unit_p_mw"].get(kd), sc["unit_e_mwh"].get(kd))
        notes["placement"] = [f"{kd}@{bu}" for kd, bu in zip(kinds, buses)]

    # ---- switchable branches
    sw = np.zeros(m, bool)
    unit_bus = {u.bus for u in units}
    deg = np.bincount(np.concatenate([d["f"], d["t"]]), minlength=n)
    nsw = sc["n_switch_lines"]
    lines = np.flatnonzero(~d["is_trafo"])
    if nsw == "all":
        sw[lines] = True
    else:
        sw[lines[np.argsort(-d["b"][lines])[: int(nsw)]]] = True   # largest charging
    if cfg["transformer"]["switchable"]:
        for k in np.flatnonzero(d["is_trafo"]):                    # not the unit step-up trafos
            gsu = (d["f"][k] in unit_bus and deg[d["f"][k]] == 1) or \
                  (d["t"][k] in unit_bus and deg[d["t"][k]] == 1)
            sw[k] = not gsu
    tap = d["tap"].copy()
    if cfg["transformer"].get("taps_nominal", False):
        notes["taps_reset"] = [float(v) for v in tap[tap != 1.0]]; tap[:] = 1.0
    gs, bs = d["gs"].copy(), d["bs"].copy()
    if not sc.get("shunts_in_service", False):
        notes["shunts_out_mvar"] = float(bs.sum()); gs[:] = 0.0; bs[:] = 0.0
    else:
        # Keep only capacitors (positive Mvar); reactors lower voltage during restoration
        reactors = bs < 0
        if reactors.any():
            notes["reactors_out_mvar"] = float(bs[reactors].sum())
            bs[reactors] = 0.0; gs[reactors] = 0.0
    return GridData(
        name=d["name"], base_mva=float(sc["base_mva"]), bus_names=d["bus_names"],
        vn_kv=d["vn_kv"], f=d["f"], t=d["t"], r=d["r"], x=d["x"], b=b, tap=tap,
        shift=d["shift"], rate=rate, is_trafo=d["is_trafo"], trafo_sn=trafo_sn,
        switchable=sw, br_names=d["br_names"], gs=gs, bs=bs, load_bus=lb, load_p=lp,
        load_q=lq, load_prio=prio, units=units, coords=d.get("coords"), notes=notes)


def describe(grid: GridData) -> dict:
    """Numbers for Table 2 / appendix of the manuscript."""
    by = lambda k: [u for u in grid.units if u.kind == k]
    return dict(
        system=grid.name, buses=grid.n_bus, branches=grid.n_br,
        lines=int((~grid.is_trafo).sum()), trafos=int(grid.is_trafo.sum()),
        switchable=int(grid.switchable.sum()), loads=grid.n_load,
        load_mw=round(grid.total_p, 3), load_mvar=round(float(grid.load_q.sum()), 3),
        charging_mvar=round(float(grid.b.sum() * grid.base_mva), 3),
        pv=[u.p_rated for u in by("PV")], wp=[u.p_rated for u in by("WP")],
        ess_mw=[u.p_rated for u in by("ESS")], ess_mwh=[u.e_mwh for u in by("ESS")],
        ibr_mw=round(sum(u.p_rated for u in grid.units), 3),
        ibr_mva=round(sum(u.s_rated for u in grid.units), 3),
        adequacy=round(sum(u.p_rated for u in grid.units) / grid.total_p, 3),
        notes=grid.notes)


if __name__ == "__main__":
    import json
    from .config import load
    cfg = load()
    for s in cfg["systems"]:
        print(json.dumps(describe(build(s, cfg)), indent=None, default=str))
