"""
Black-Start Restoration Gymnasium Environment
==============================================
Fully functional OpenAI-Gym-compatible environment that wraps
pandapower networks for sequential power-system restoration.

The agent controls:
  - switch closures   (discrete: which switch to close, or no-op)
  - generator dispatch (continuous: active-power set-points)
  - load pickup        (discrete: energise a load block or not)

Observations include bus voltages, line loadings, generator states,
ESS state-of-charge, and physics-informed features (power imbalance,
estimated SCR, RoCoF proxy, voltage deviation).

The environment keeps a JSON-serialisable log of every step so it
can be replayed in the HTML visualiser.
"""

from __future__ import annotations
import json, copy, math, random, warnings, logging, os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ['NUMBA_DISABLE_JIT'] = '1'
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import gymnasium as gym
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import networkx as nx
from gymnasium import spaces


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialisation."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# Helper: build modified IEEE test cases with DER + switches
# ---------------------------------------------------------------------------

def _build_ieee14_blackstart() -> pp.pandapowerNet:
    """Modified IEEE-14 with GFM BESS, PV, Wind, ESS, extra switches."""
    net = pn.case14()

    # Remove all existing generators (we replace them with IBR)
    net.gen.drop(net.gen.index, inplace=True)
    net.sgen.drop(net.sgen.index, inplace=True)

    # ----- GFM BESS black-start unit at bus 0 -----
    pp.create_sgen(net, bus=0, p_mw=0.0, q_mvar=0.0, name="GFM_BESS_0",
                   max_p_mw=90, min_p_mw=-90, max_q_mvar=40, min_q_mvar=-40,
                   controllable=True, in_service=True)

    # ----- GFM PV at bus 1 -----
    pp.create_sgen(net, bus=1, p_mw=0.0, q_mvar=0.0, name="GFM_PV_1",
                   max_p_mw=50, min_p_mw=0, max_q_mvar=25, min_q_mvar=-25,
                   controllable=True, in_service=False)

    # ----- GFM Wind at bus 2 -----
    pp.create_sgen(net, bus=2, p_mw=0.0, q_mvar=0.0, name="GFM_Wind_2",
                   max_p_mw=40, min_p_mw=0, max_q_mvar=20, min_q_mvar=-20,
                   controllable=True, in_service=False)

    # ----- GFL Solar at bus 5, 7 -----
    for b in [5, 7]:
        pp.create_sgen(net, bus=b, p_mw=0.0, q_mvar=0.0,
                       name=f"GFL_Solar_{b}", max_p_mw=30, min_p_mw=0,
                       controllable=True, in_service=False)

    # ----- Mark all loads out-of-service (blackout) -----
    net.load["in_service"] = False

    # ----- Add controllable switches on selected lines -----
    # We add a switch on every line; closed=False means blackout state
    for idx in net.line.index:
        pp.create_switch(net, bus=net.line.at[idx, "from_bus"],
                         element=idx, et="l", closed=False,
                         name=f"sw_line_{idx}")

    # Store original load values for later reference
    net["load_p_mw_orig"] = net.load["p_mw"].values.copy()
    net["load_q_mvar_orig"] = net.load["q_mvar"].values.copy()

    return net


# ---------------------------------------------------------------------------
# ESS book-keeping (lightweight, no extra package)
# ---------------------------------------------------------------------------

class ESSState:
    """Track SOC for one battery."""
    def __init__(self, bus: int, energy_mwh: float, p_max_mw: float,
                 eta: float = 0.95, soc_init: float = 0.7):
        self.bus = bus
        self.energy_mwh = energy_mwh
        self.p_max_mw = p_max_mw
        self.eta = eta
        self.soc = soc_init

    def step(self, p_mw: float, dt_h: float = 1 / 60):
        """Advance SOC.  p_mw > 0 → discharge."""
        p_mw = float(np.clip(p_mw, -self.p_max_mw, self.p_max_mw))
        if p_mw > 0:
            de = p_mw * dt_h / self.eta
        else:
            de = p_mw * dt_h * self.eta        # negative → soc increases
        self.soc = float(np.clip(self.soc - de / self.energy_mwh, 0.05, 0.95))
        return self.soc

    def reset(self, soc: float | None = None):
        self.soc = soc if soc is not None else random.uniform(0.3, 0.9)


# ---------------------------------------------------------------------------
# Main Environment
# ---------------------------------------------------------------------------

class BlackStartRestorationEnv(gym.Env):
    """
    Gymnasium environment for sequential black-start restoration.

    Action space (MultiDiscrete):
        [switch_action, gen0_pwr, gen1_pwr, gen2_pwr, load_action]
        - switch_action : 0..n_switches  (last = no-op)
        - gen*_pwr      : 0..10  discretised power level
        - load_action   : 0..n_loads     (last = no-op)

    Observation space (Box):
        Flat vector of bus voltages, line loadings, gen outputs,
        ESS soc, switch states, load states, physics features.
    """

    metadata = {"render_modes": ["human", "json"]}

    # ---- construction ----
    def __init__(self, render_mode: str | None = None, max_steps: int = 20,
                 network: str = "ieee14"):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.network_name = network

        # Build network
        self.net = _build_ieee14_blackstart()  # extend for 39/123 later
        self._snapshot = copy.deepcopy(self.net)  # pristine copy for reset

        # Counts
        self.n_bus = len(self.net.bus)
        self.n_line = len(self.net.line)
        self.n_switch = len(self.net.switch)
        self.n_gen = len(self.net.sgen)
        self.n_load = len(self.net.load)

        # ESS (co-located with GFM BESS at bus 0)
        self.ess = ESSState(bus=0, energy_mwh=200, p_max_mw=90, soc_init=0.7)

        # Priority weights per load (random assignment for demo)
        rng = np.random.RandomState(42)
        self.load_priorities = rng.choice([1.0, 2.0, 3.0], size=self.n_load)

        # ----- action space -----
        self.n_gen_levels = 11  # 0..10 → maps to 0..p_max in 10% steps
        self.action_space = spaces.MultiDiscrete([
            self.n_switch + 1,     # switch to close (last = noop)
            self.n_gen_levels,     # GFM BESS power level
            self.n_gen_levels,     # GFM PV power level
            self.n_gen_levels,     # GFM Wind power level
            self.n_load + 1,       # load to pick up (last = noop)
        ])

        # ----- observation space -----
        # bus_v(n_bus) + line_loading(n_line) + sw_state(n_switch)
        # + gen_p(n_gen) + load_state(n_load) + ess_soc(1) + freq(1)
        # + physics(5)
        self.obs_dim = (self.n_bus + self.n_line + self.n_switch
                        + self.n_gen + self.n_load + 1 + 1 + 5)
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(self.obs_dim,), dtype=np.float32)

        # ----- episode bookkeeping -----
        self.step_count = 0
        self.energised_buses: set[int] = set()
        self.restored_loads: set[int] = set()
        self.frequency = 50.0
        self.episode_log: list[dict] = []  # for HTML visualiser
        self.total_reward = 0.0
        self._prev_lrr = 0.0

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.net = copy.deepcopy(self._snapshot)
        self.step_count = 0
        self.energised_buses = {0}          # black-start bus energised
        self.restored_loads = set()
        self.frequency = 50.0
        self.total_reward = 0.0
        self._prev_lrr = 0.0
        self.ess.reset()
        self.episode_log = []

        # randomise load levels slightly
        scale = self.np_random.uniform(0.7, 1.0)
        self.net.load["p_mw"] = self.net["load_p_mw_orig"] * scale
        self.net.load["q_mvar"] = self.net["load_q_mvar_orig"] * scale

        # Black-start gen is already in-service
        self.net.sgen.at[0, "in_service"] = True

        obs = self._get_obs()
        info = self._get_info()
        self._log_step("reset", {})
        return obs, info

    # ------------------------------------------------------------------- step
    def step(self, action):
        self.step_count += 1
        act_sw, act_g0, act_g1, act_g2, act_ld = int(action[0]), int(action[1]), int(action[2]), int(action[3]), int(action[4])
        action_record = {"switch": act_sw, "gens": [act_g0, act_g1, act_g2],
                         "load": act_ld}

        # ---- 1. Switch closure ----
        if act_sw < self.n_switch:
            sw = self.net.switch.iloc[act_sw]
            line_idx = sw["element"]
            from_b = self.net.line.at[line_idx, "from_bus"]
            to_b = self.net.line.at[line_idx, "to_bus"]
            # only close if at least one end is energised
            if from_b in self.energised_buses or to_b in self.energised_buses:
                self.net.switch.at[act_sw, "closed"] = True
                self.net.line.at[line_idx, "in_service"] = True
                self.energised_buses.update([from_b, to_b])
                # flood-fill energised buses via closed switches
                self._flood_energise()

        # ---- 2. Generator dispatch ----
        gen_setpoints = [act_g0, act_g1, act_g2]
        for gi, lvl in enumerate(gen_setpoints):
            if gi < self.n_gen:
                p_max = self.net.sgen.at[gi, "max_p_mw"]
                p_set = p_max * lvl / (self.n_gen_levels - 1)
                self.net.sgen.at[gi, "p_mw"] = p_set
                bus = self.net.sgen.at[gi, "bus"]
                if bus in self.energised_buses:
                    self.net.sgen.at[gi, "in_service"] = True

        # ---- 3. Load pickup ----
        if act_ld < self.n_load:
            ld_bus = self.net.load.at[act_ld, "bus"]
            if ld_bus in self.energised_buses and act_ld not in self.restored_loads:
                self.net.load.at[act_ld, "in_service"] = True
                self.restored_loads.add(act_ld)

        # ---- 4. ESS step ----
        ess_p = self.net.sgen.at[0, "p_mw"]  # BESS power
        self.ess.step(ess_p)

        # ---- 5. Run power flow ----
        converged = self._run_pf()

        # ---- 6. Compute frequency proxy (droop) ----
        if converged:
            total_gen = self.net.res_sgen.p_mw.sum()
            total_load = self.net.res_load.p_mw.sum()
            dp = (total_load - total_gen)
            total_cap = self.net.sgen.max_p_mw.sum()
            self.frequency = 50.0 - 0.03 * (dp / max(total_cap, 1)) * 50
        else:
            self.frequency = 50.0

        # ---- 7. Reward ----
        reward, info_r = self._compute_reward(converged)
        self.total_reward += reward

        # ---- 8. Termination ----
        terminated = False
        if not converged:
            terminated = True
            reward -= 5.0
        if abs(self.frequency - 50) > 3:
            terminated = True
            reward -= 5.0
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()
        info.update(info_r)
        info["converged"] = converged

        self._log_step("step", action_record)
        return obs, reward, terminated, truncated, info

    # -------------------------------------------------------------- internals
    def _flood_energise(self):
        """BFS from energised buses through closed switches."""
        G = nx.Graph()
        for idx in self.net.switch.index:
            if self.net.switch.at[idx, "closed"]:
                li = self.net.switch.at[idx, "element"]
                f = self.net.line.at[li, "from_bus"]
                t = self.net.line.at[li, "to_bus"]
                G.add_edge(f, t)
        for b in list(self.energised_buses):
            if G.has_node(b):
                self.energised_buses.update(nx.node_connected_component(G, b))

    def _run_pf(self) -> bool:
        try:
            pp.runpp(self.net, algorithm="nr", max_iteration=30,
                     enforce_q_lims=False, calculate_voltage_angles=True)
            return bool(self.net.converged)
        except Exception:
            return False

    def _compute_reward(self, converged: bool):
        """Multi-component physics-informed reward."""
        info = {}
        if not converged:
            return -2.0, info

        # -- load recovery delta --
        total_load = self.net.load.p_mw.sum()
        rest_load = self.net.res_load.p_mw[self.net.load.in_service].sum()
        lrr = rest_load / max(total_load, 1e-6)
        delta_lrr = lrr - self._prev_lrr
        self._prev_lrr = lrr
        r_restore = 20.0 * delta_lrr

        # -- voltage --
        v = self.net.res_bus.vm_pu.values
        v_viol = np.sum((v < 0.9) | (v > 1.1))
        r_volt = -1.0 * v_viol

        # -- frequency --
        fd = abs(self.frequency - 50)
        r_freq = 0.1 if fd < 0.5 else -fd

        # -- power imbalance --
        dp = abs(self.net.res_sgen.p_mw.sum() - self.net.res_load.p_mw.sum())
        r_phys = -0.5 * dp / max(self.net.sgen.max_p_mw.sum(), 1)

        # -- line overload --
        loading = self.net.res_line.loading_percent.values / 100
        r_line = -1.0 * np.sum(loading > 1.0)

        reward = float(r_restore + r_volt + r_freq + r_phys + r_line)
        info.update({"lrr": lrr, "freq": self.frequency,
                     "v_viol": int(v_viol), "dp_pu": float(dp)})
        return reward, info

    def _get_obs(self) -> np.ndarray:
        parts = []
        # bus voltages
        if hasattr(self.net, "res_bus") and len(self.net.res_bus) == self.n_bus:
            parts.append(self.net.res_bus.vm_pu.values.astype(np.float32))
        else:
            v = np.zeros(self.n_bus, dtype=np.float32)
            for b in self.energised_buses:
                if b < self.n_bus:
                    v[b] = 1.0
            parts.append(v)

        # line loading
        if hasattr(self.net, "res_line") and len(self.net.res_line) == self.n_line:
            parts.append((self.net.res_line.loading_percent.values / 100).astype(np.float32))
        else:
            parts.append(np.zeros(self.n_line, dtype=np.float32))

        # switch states
        parts.append(self.net.switch.closed.values.astype(np.float32))

        # gen active power (pu of max)
        gen_pu = np.zeros(self.n_gen, dtype=np.float32)
        for gi in range(self.n_gen):
            mx = self.net.sgen.at[gi, "max_p_mw"]
            if mx > 0 and self.net.sgen.at[gi, "in_service"]:
                gen_pu[gi] = self.net.sgen.at[gi, "p_mw"] / mx
        parts.append(gen_pu)

        # load states
        parts.append(self.net.load.in_service.values.astype(np.float32))

        # ESS soc
        parts.append(np.array([self.ess.soc], dtype=np.float32))

        # frequency
        parts.append(np.array([self.frequency / 50.0], dtype=np.float32))

        # physics features (5): dp, dq, scr_est, rocof_proxy, dv
        total_gen = self.net.res_sgen.p_mw.sum() if self._run_ok() else 0
        total_load = self.net.res_load.p_mw.sum() if self._run_ok() else 0
        dp = (total_load - total_gen) / max(self.net.sgen.max_p_mw.sum(), 1)
        dq = 0.0
        scr = len(self.energised_buses) / max(self.n_bus, 1)
        rocof = (self.frequency - 50) / max(self.step_count, 1)
        dv = float(np.mean(self.net.res_bus.vm_pu.values) - 1.0) if self._run_ok() else 0
        parts.append(np.array([dp, dq, scr, rocof, dv], dtype=np.float32))

        obs = np.concatenate(parts)
        # pad / clip to obs_dim
        if len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)))
        return obs[:self.obs_dim]

    def _run_ok(self):
        return hasattr(self.net, "res_bus") and len(self.net.res_bus) == self.n_bus

    def _get_info(self) -> dict:
        return {
            "step": self.step_count,
            "energised_buses": sorted(self.energised_buses),
            "restored_loads": sorted(self.restored_loads),
            "ess_soc": self.ess.soc,
            "frequency": self.frequency,
            "n_switches_closed": int(self.net.switch.closed.sum()),
        }

    # --------------------------------------------------------- logging for viz
    def _log_step(self, event: str, action: dict):
        entry = {
            "event": event,
            "step": self.step_count,
            "action": action,
            "energised_buses": sorted(self.energised_buses),
            "restored_loads": sorted(self.restored_loads),
            "switch_states": self.net.switch.closed.values.tolist(),
            "ess_soc": round(self.ess.soc, 4),
            "frequency": round(self.frequency, 4),
            "total_reward": round(self.total_reward, 4),
        }
        # bus voltages if available
        if self._run_ok():
            entry["bus_voltages"] = [round(v, 4) for v in self.net.res_bus.vm_pu.values]
            entry["gen_p_mw"] = [round(p, 2) for p in self.net.res_sgen.p_mw.values]
            entry["load_p_mw"] = [round(p, 2) for p in self.net.res_load.p_mw.values]
            entry["line_loading"] = [round(l, 2) for l in self.net.res_line.loading_percent.values]
            entry["lrr"] = round(
                self.net.res_load.p_mw[self.net.load.in_service].sum()
                / max(self.net.load.p_mw.sum(), 1e-6), 4)
        self.episode_log.append(entry)

    def save_episode_log(self, path: str):
        """Save episode replay to JSON for the HTML visualiser."""
        # also save network topology (bus coords + lines) for drawing
        topo = {
            "buses": [],
            "lines": [],
            "gens": [],
            "loads": [],
        }
        # bus positions: use pandapower plotting coords or make a circle layout
        try:
            from pandapower.plotting import create_generic_coordinates
            create_generic_coordinates(self.net)
            for bi in self.net.bus.index:
                topo["buses"].append({
                    "id": int(bi),
                    "x": float(self.net.bus_geodata.at[bi, "x"]),
                    "y": float(self.net.bus_geodata.at[bi, "y"]),
                    "name": self.net.bus.at[bi, "name"] if "name" in self.net.bus.columns else f"Bus {bi}",
                })
        except Exception:
            # fallback: circular layout
            for i, bi in enumerate(self.net.bus.index):
                angle = 2 * math.pi * i / self.n_bus
                topo["buses"].append({
                    "id": int(bi), "x": 300 + 200 * math.cos(angle),
                    "y": 300 + 200 * math.sin(angle),
                    "name": f"Bus {bi}",
                })

        for li in self.net.line.index:
            topo["lines"].append({
                "id": int(li),
                "from": int(self.net.line.at[li, "from_bus"]),
                "to": int(self.net.line.at[li, "to_bus"]),
            })
        for gi in self.net.sgen.index:
            topo["gens"].append({
                "id": int(gi), "bus": int(self.net.sgen.at[gi, "bus"]),
                "name": self.net.sgen.at[gi, "name"],
                "max_p": float(self.net.sgen.at[gi, "max_p_mw"]),
            })
        for li in self.net.load.index:
            topo["loads"].append({
                "id": int(li), "bus": int(self.net.load.at[li, "bus"]),
                "p_mw": float(self.net.load.at[li, "p_mw"]),
            })

        payload = {"topology": topo, "steps": self.episode_log}
        Path(path).write_text(json.dumps(payload, indent=2, cls=_NumpyEncoder))

    def render(self):
        if self.render_mode == "human":
            info = self._get_info()
            print(f"  Step {info['step']:2d} | "
                  f"buses={len(info['energised_buses']):2d}/{self.n_bus} | "
                  f"loads={len(info['restored_loads']):2d}/{self.n_load} | "
                  f"soc={info['ess_soc']:.2f} | "
                  f"freq={info['frequency']:.2f} Hz | "
                  f"reward={self.total_reward:.2f}")
