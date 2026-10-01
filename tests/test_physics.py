"""Power-flow, frequency-model and network-data checks (no learning involved)."""
import logging
import warnings

import numpy as np
import pytest

from sim.acpf import DroopPowerFlow, UnitState
from sim.config import load
from sim.frequency import FrequencyModel
from sim.networks import build, describe

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
CFG = load()


def _full(g, scale):
    pl = np.zeros(g.n_bus); ql = np.zeros(g.n_bus)
    np.add.at(pl, g.load_bus, scale * g.load_p / g.base_mva); np.add.at(ql, g.load_bus, scale * g.load_q / g.base_mva)
    return np.ones(g.n_bus, bool), np.ones(g.n_br, bool), pl, ql


def test_droop_pf_matches_pandapower():
    import pandapower as pp, pandapower.networks as pn
    cfg = load(overrides={"transformer": {"taps_nominal": False}, "systems": {"14": {"shunts_in_service": False}}})
    g = build("14", cfg); pf = DroopPowerFlow(g)
    bus_on, br_on, pl, ql = _full(g, 0.5)
    units = [UnitState(bus=u.bus, s=u.s_rated / g.base_mva, p_ref=0.0, p_min=-9, p_max=9, i_max=99) for u in g.units]
    r = pf.solve(bus_on, br_on, pl, ql, units)
    assert r.ok
    net = pn.case14(); net.gen.drop(net.gen.index, inplace=True); net.ext_grid.drop(net.ext_grid.index, inplace=True)
    net.shunt["in_service"] = False; net.load["p_mw"] *= 0.5; net.load["q_mvar"] *= 0.5
    for k, u in enumerate(g.units):
        pp.create_gen(net, bus=u.bus, p_mw=r.p_unit[k] * 100, vm_pu=r.v[u.bus], slack=(k == 0))
    pp.runpp(net)
    assert np.max(np.abs(net.res_bus.vm_pu.values - r.v)) < 1e-6
    assert np.allclose(net.res_gen.q_mvar.values, r.q_unit * 100, atol=1e-3)


def test_droop_laws_and_sign():
    g = build("14", CFG); pf = DroopPowerFlow(g)
    bus_on, br_on, pl, ql = _full(g, 0.4)
    units = [UnitState(bus=u.bus, s=u.s_rated / g.base_mva, p_ref=0.0, p_min=-9, p_max=9, i_max=99) for u in g.units]
    r = pf.solve(bus_on, br_on, pl, ql, units)
    s = np.array([u.s for u in units])
    assert np.allclose(r.p_unit / s, (r.p_unit / s)[0], atol=1e-8)          # proportional sharing (Eq. 9)
    assert np.allclose(r.v[[u.bus for u in units]], 1.0 - 0.04 * r.q_unit / s, atol=1e-7)   # Q-V droop (Eq. 10)
    assert r.dw[0] < 0                                                       # deficit -> under-frequency
    assert abs(r.dw[0] + CFG["gfm"]["droop_p"] * r.p_unit.sum() / s.sum()) < 1e-8
    assert abs(r.p_unit.sum() - pl.sum() - r.losses) < 1e-6                  # power balance


def test_unit_limit_clamps_and_others_pick_up():
    g = build("14", CFG); pf = DroopPowerFlow(g)
    bus_on, br_on, pl, ql = _full(g, 0.4)
    units = [UnitState(bus=u.bus, s=u.s_rated / g.base_mva, p_ref=0.0, p_min=0.0, p_max=9, i_max=99) for u in g.units]
    units[0].p_max = 0.05
    r = pf.solve(bus_on, br_on, pl, ql, units)
    assert r.ok and r.p_clamped[0] and abs(r.p_unit[0] - 0.05) < 1e-9
    assert abs(r.p_unit.sum() - pl.sum() - r.losses) < 1e-6


def test_two_islands_have_independent_frequencies():
    g = build("14", CFG); pf = DroopPowerFlow(g)
    bus_on = np.zeros(g.n_bus, bool); bus_on[[0, 2]] = True
    pl = np.zeros(g.n_bus); ql = np.zeros(g.n_bus); pl[2] = 0.1
    units = [UnitState(bus=0, s=0.5, p_ref=0.0, p_max=9), UnitState(bus=2, s=0.8, p_ref=0.0, p_max=9)]
    r = pf.solve(bus_on, np.zeros(g.n_br, bool), pl, ql, units)
    assert r.ok and r.island[0] != r.island[2]
    assert abs(r.dw[0]) < 1e-12 and r.dw[2] < 0


def test_frequency_model_closed_forms():
    fm = FrequencyModel(CFG); gcfg = CFG["gfm"]
    ro, nad, st = fm.event(10.0, 100.0)
    assert ro == pytest.approx(-0.1 * 50 / (2 * gcfg["h_s"]))
    assert st == pytest.approx(-0.1 * 50 / (gcfg["d_pu"] + 1 / gcfg["droop_p"]))
    assert nad <= st < 0                                                     # nadir below settling value
    step = fm.max_step_mw(100.0, 0.0, 1.0, 0.5)
    ro2, nad2, _ = fm.event(step, 100.0)
    assert abs(ro2) <= 1.0 + 1e-9 and abs(nad2) <= 0.5 + 1e-9
    assert fm.max_step_mw(100.0, -0.3, 1.0, 0.5) <= step                     # less room when already low


def test_networks_match_documented_modifications():
    d14, d39, d118 = (describe(build(s, CFG)) for s in ("14", "39", "118"))
    assert (d14["buses"], d14["loads"], d14["switchable"]) == (14, 11, 20) and d14["load_mw"] == pytest.approx(259.0)
    assert d14["adequacy"] == pytest.approx(1.100, abs=1e-3)
    # With shunts_in_service=true, charging_mvar includes bus shunts
    assert (d39["buses"], d39["loads"], d39["switchable"]) == (39, 28, 20)
    assert d39["adequacy"] == pytest.approx(1.125, abs=2e-3)
    assert (d118["buses"], d118["loads"], d118["switchable"]) == (118, 99, 98)
    assert d118["adequacy"] == pytest.approx(1.180, abs=2e-3)
    assert all(d["adequacy"] > 1.05 for d in (d14, d39, d118))             # sufficient for LRR > 0.9
