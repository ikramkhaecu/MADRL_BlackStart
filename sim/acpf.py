"""Droop-based AC power flow for inverter-formed islands.

During a black start there is no infinite bus.  Every energised island is
formed by its grid-forming inverters, so the steady state is fixed by their
primary controllers (manuscript Eq. 9-10):

    P_k = P_ref,k - (S_k / m_p) * dw          (P-f droop, dw in pu)
    Q_k = Q_ref,k + (S_k / n_q) * (V0 - V_k)  (Q-V droop)

The solver therefore has *no slack bus*: the unknowns are the bus angles
(one reference angle per island), all bus voltage magnitudes and the island
frequency deviation dw; the equations are the 2N nodal balances.  Unit limits
(available power, SoC-dependent ESS limits, converter current) are enforced
by clamping a unit to fixed P and/or fixed Q and re-solving.  The converged
Jacobian is kept so that the environment can predict the effect of a load
pick-up on every bus voltage and on the frequency without another solve
(this is what the physics-informed action mask uses).

v1 kept the case's ``ext_grid`` in service, i.e. an infinite bus supplied any
imbalance at 1.0 pu; GFM capability could therefore never bind.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class UnitState:
    bus: int
    s: float                 # MVA rating (pu on system base)
    p_ref: float             # pu
    q_ref: float = 0.0
    v0: float = 1.0
    mp: float = 0.03
    nq: float = 0.04
    p_min: float = 0.0
    p_max: float = 1e9
    i_max: float = 1.2       # pu of unit rating


@dataclass
class PFResult:
    ok: bool
    reason: str
    v: np.ndarray            # |V| per bus (0 on dead buses)
    th: np.ndarray
    dw: np.ndarray           # island frequency deviation per bus (pu)
    island: np.ndarray       # island id per bus (-1 dead)
    p_unit: np.ndarray
    q_unit: np.ndarray
    p_clamped: np.ndarray
    q_clamped: np.ndarray
    s_from: np.ndarray       # complex branch flow at from side (pu)
    s_to: np.ndarray
    losses: float
    n_iter: int
    sens: dict               # island id -> dict(buses, Jinv, nb)


class DroopPowerFlow:
    def __init__(self, grid):
        self.g = grid
        ys = 1.0 / (grid.r + 1j * grid.x)
        bc = 1j * grid.b / 2.0
        tap = grid.tap * np.exp(1j * grid.shift)
        self.ytt = ys + bc
        self.yff = self.ytt / (tap * np.conj(tap))
        self.yft = -ys / np.conj(tap)
        self.ytf = -ys / tap
        self.ysh = (grid.gs + 1j * grid.bs) / grid.base_mva

    # ------------------------------------------------------------- topology
    def islands(self, bus_on, br_on):
        g = self.g
        lab = np.full(g.n_bus, -1)
        adj = [[] for _ in range(g.n_bus)]
        for k in np.flatnonzero(br_on):
            a, b = g.f[k], g.t[k]
            if bus_on[a] and bus_on[b]:
                adj[a].append(b); adj[b].append(a)
        c = 0
        for s in np.flatnonzero(bus_on):
            if lab[s] >= 0:
                continue
            lab[s] = c; st = [s]
            while st:
                u = st.pop()
                for v in adj[u]:
                    if lab[v] < 0:
                        lab[v] = c; st.append(v)
            c += 1
        return lab, c

    def _ybus(self, buses, br_on):
        g, nb = self.g, len(buses)
        idx = -np.ones(g.n_bus, int); idx[buses] = np.arange(nb)
        k = np.flatnonzero(br_on & (idx[g.f] >= 0) & (idx[g.t] >= 0))
        a, b = idx[g.f[k]], idx[g.t[k]]
        Y = np.zeros((nb, nb), complex)
        np.add.at(Y, (a, a), self.yff[k]); np.add.at(Y, (b, b), self.ytt[k])
        np.add.at(Y, (a, b), self.yft[k]); np.add.at(Y, (b, a), self.ytf[k])
        Y[np.arange(nb), np.arange(nb)] += self.ysh[buses]
        return Y, idx

    # ---------------------------------------------------------------- solve
    def solve(self, bus_on, br_on, p_load, q_load, units, v_init=None, th_init=None,
              tol=1e-7, max_iter=25, need_sens=True):
        """bus_on/br_on: bool arrays; p_load/q_load: pu per bus; units: list[UnitState]."""
        g = self.g
        n = g.n_bus
        V = np.zeros(n); TH = np.zeros(n); DW = np.zeros(n)
        nu = len(units)
        PU = np.zeros(nu); QU = np.zeros(nu)
        PC = np.zeros(nu, bool); QC = np.zeros(nu, bool)
        lab, nis = self.islands(bus_on, br_on)
        sens, tot_it = {}, 0
        ok, reason = True, ""
        for c in range(nis):
            buses = np.flatnonzero(lab == c)
            uidx = [k for k, u in enumerate(units) if lab[u.bus] == c]
            if not uidx:
                lab[buses] = -1                       # no source -> dead island
                if p_load[buses].sum() > 1e-9:
                    ok, reason = False, "island_without_source"
                continue
            res = self._solve_island(buses, br_on, p_load, q_load, [units[k] for k in uidx],
                                     v_init, th_init, tol, max_iter, need_sens)
            if not res["ok"] and res["reason"] in ("voltage_collapse", "pf_not_converged",
                                                   "singular_jacobian"):
                res2 = self._solve_island(buses, br_on, p_load, q_load, [units[k] for k in uidx],
                                          None, None, tol, max_iter, need_sens)   # flat-start retry
                res2["it"] += res["it"]
                res = res2
            tot_it += res["it"]
            if not res["ok"]:
                ok, reason = False, res["reason"]
                V[buses] = np.nan
                continue
            V[buses], TH[buses], DW[buses] = res["v"], res["th"], res["dw"]
            PU[uidx], QU[uidx] = res["p"], res["q"]
            PC[uidx], QC[uidx] = res["pc"], res["qc"]
            sens[c] = dict(buses=buses, Jinv=res["Jinv"], nb=len(buses), ref=res["ref"],
                           units=uidx, kq=res["kq"], kp=res["kp"])
        Vc = np.nan_to_num(V) * np.exp(1j * TH)
        live = br_on & (lab[g.f] >= 0) & (lab[g.t] >= 0)
        sf = np.zeros(g.n_br, complex); st = np.zeros(g.n_br, complex)
        k = np.flatnonzero(live)
        i_f = self.yff[k] * Vc[g.f[k]] + self.yft[k] * Vc[g.t[k]]
        i_t = self.ytf[k] * Vc[g.f[k]] + self.ytt[k] * Vc[g.t[k]]
        sf[k] = Vc[g.f[k]] * np.conj(i_f); st[k] = Vc[g.t[k]] * np.conj(i_t)
        return PFResult(ok, reason, V, TH, DW, lab, PU, QU, PC, QC, sf, st,
                        float((sf + st).real.sum()), tot_it, sens)

    def _solve_island(self, buses, br_on, p_load, q_load, units, v_init, th_init, tol, max_iter,
                      need_sens=True):
        nb = len(buses)
        Y, idx = self._ybus(buses, br_on)
        ub = np.array([idx[u.bus] for u in units])
        s = np.array([u.s for u in units])
        kp_all = s / np.array([u.mp for u in units])
        kq_all = s / np.array([u.nq for u in units])
        pref = np.array([u.p_ref for u in units]); qref = np.array([u.q_ref for u in units])
        v0 = np.array([u.v0 for u in units])
        pmin = np.array([u.p_min for u in units]); pmax = np.array([u.p_max for u in units])
        imax = np.array([u.i_max for u in units])
        pc = np.zeros(len(units), bool); qc = np.zeros(len(units), bool)
        pfix = np.zeros(len(units)); qfix = np.zeros(len(units))
        pl, ql = p_load[buses], q_load[buses]
        if v_init is None:
            v, th = np.ones(nb), np.zeros(nb)
        else:                       # warm start; newly energised buses (NaN) take the island mean angle
            known = np.isfinite(v_init[buses]) & (v_init[buses] > 0.5)
            v = np.where(known, v_init[buses], 1.0)
            th0 = th_init[buses] - th_init[buses[ub[0]]]
            th = np.where(known, th0, th0[known].mean() if known.any() else 0.0)
        ref = int(ub[0]); nonref = np.array([i for i in range(nb) if i != ref], int)
        dw, it_total = 0.0, 0
        for _outer in range(8):
            free_p, free_q = ~pc, ~qc
            if not free_p.any():
                return dict(ok=False, reason="generation_inadequate", it=it_total)
            kp = np.where(free_p, kp_all, 0.0); kq = np.where(free_q, kq_all, 0.0)
            KP = np.zeros(nb); np.add.at(KP, ub, kp)
            KQ = np.zeros(nb); np.add.at(KQ, ub, kq)
            conv = False
            for it in range(max_iter):
                Vc = v * np.exp(1j * th)
                Ibus = Y @ Vc
                S = Vc * np.conj(Ibus)
                pu = np.where(free_p, pref - kp_all * dw, pfix)
                qu = np.where(free_q, qref + kq_all * (v0 - v[ub]), qfix)
                Pg = np.zeros(nb); np.add.at(Pg, ub, pu)
                Qg = np.zeros(nb); np.add.at(Qg, ub, qu)
                F = np.concatenate([S.real - (Pg - pl), S.imag - (Qg - ql)])
                if np.max(np.abs(F)) < tol:
                    conv = True
                    break
                J = self._jac(Y, Vc, Ibus, th, nb, nonref, KQ, KP)
                try:
                    dx = np.linalg.solve(J, -F)
                except np.linalg.LinAlgError:
                    return dict(ok=False, reason="singular_jacobian", it=it_total + it)
                th[nonref] += dx[:nb - 1]; v += dx[nb - 1:2 * nb - 1]; dw += dx[2 * nb - 1]
                if not np.all(np.isfinite(v)) or v.min() < 0.3 or v.max() > 2.0:
                    return dict(ok=False, reason="voltage_collapse", it=it_total + it)
            it_total += it + 1
            if not conv:
                return dict(ok=False, reason="pf_not_converged", it=it_total)
            pu = np.where(free_p, pref - kp_all * dw, pfix)
            qu = np.where(free_q, qref + kq_all * (v0 - v[ub]), qfix)
            changed = False
            hi, lo = free_p & (pu > pmax + 1e-9), free_p & (pu < pmin - 1e-9)
            if hi.any() or lo.any():
                pc |= hi | lo; pfix = np.where(hi, pmax, np.where(lo, pmin, pfix)); changed = True
            else:
                pnow = np.where(pc, pfix, pu)
                qcap = np.sqrt(np.maximum((imax * s * v[ub]) ** 2 - pnow ** 2, 0.0))
                over = free_q & (np.abs(qu) > qcap + 1e-9)
                if over.any():
                    qc |= over; qfix = np.where(over, np.sign(qu) * qcap, qfix); changed = True
                # release P clamps that are no longer needed is deliberately NOT done
            if not changed:
                break
        else:
            return dict(ok=False, reason="limit_cycling", it=it_total)
        if not (~qc).any():
            return dict(ok=False, reason="no_voltage_regulation", it=it_total)
        Jinv = None
        if need_sens:
            Vc = v * np.exp(1j * th)
            try:
                Jinv = np.linalg.inv(self._jac(Y, Vc, Y @ Vc, th, nb, nonref, KQ, KP))
            except np.linalg.LinAlgError:
                Jinv = None
        return dict(ok=True, reason="", v=v, th=th, dw=np.full(nb, dw), p=pu, q=qu, pc=pc,
                    qc=qc, Jinv=Jinv, it=it_total, ref=ref, kq=kq, kp=kp)

    @staticmethod
    def _jac(Y, Vc, Ibus, th, nb, nonref, KQ, KP):
        En = np.exp(1j * th)
        dVa = 1j * Vc[:, None] * np.conj(np.diag(Ibus) - Y * Vc[None, :])
        dVm = Vc[:, None] * np.conj(Y * En[None, :])
        dVm[np.arange(nb), np.arange(nb)] += np.conj(Ibus) * En
        J = np.empty((2 * nb, 2 * nb))
        J[:nb, :nb - 1] = dVa.real[:, nonref]; J[nb:, :nb - 1] = dVa.imag[:, nonref]
        J[:nb, nb - 1:2 * nb - 1] = dVm.real
        J[nb:, nb - 1:2 * nb - 1] = dVm.imag
        J[nb + np.arange(nb), nb - 1 + np.arange(nb)] += KQ
        J[:nb, 2 * nb - 1] = KP; J[nb:, 2 * nb - 1] = 0.0
        return J

    # -------------------------------------------------------- sensitivities
    @staticmethod
    def predict_pickup(sens_c, bus_local, dp, dq):
        """First-order change of (V at every island bus, dw) for an extra load
        (dp, dq) [pu] at local bus index ``bus_local`` of island ``sens_c``."""
        nb, Jinv = sens_c["nb"], sens_c["Jinv"]
        if Jinv is None:
            return None, None
        dx = -(Jinv[:, bus_local] * dp + Jinv[:, nb + bus_local] * dq)      # two non-zeros only
        return dx[nb - 1:2 * nb - 1], dx[2 * nb - 1]
