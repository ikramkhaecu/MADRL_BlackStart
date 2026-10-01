"""Aggregated frequency response of a GFM-formed island (manuscript Eq. 9, 11).

Every grid-forming unit is a virtual synchronous machine with inertia constant
H, damping D and a P-f droop m_p whose power measurement passes a first-order
filter T_f (droop + filter is equivalent to VSM inertia, ref. [12] of the
manuscript).  With identical per-unit parameters on each unit's own MVA base,
the island's centre-of-inertia response to a load step dp (pu of the MVA of
the units that still have headroom) is

    2H d(df)/dt = -dp - D df - y ,      T_f dy/dt = df / m_p - y

so that   RoCoF(0+) = -dp f0 / 2H ,   df(inf) = -dp / (D + 1/m_p),
and the nadir follows from the second-order response, which is computed once
for dp = 1 and scaled.  v1 integrated ``2H df/dt = dp - D df`` for 0.2 s on
the *system-load* base with no droop, so its "nadir" was just 0.2 s x RoCoF.
"""
from __future__ import annotations

import numpy as np


class FrequencyModel:
    def __init__(self, cfg):
        g = cfg["gfm"]
        self.f0 = float(g["f_nom_hz"])
        self.H, self.D, self.mp, self.Tf = float(g["h_s"]), float(g["d_pu"]), float(g["droop_p"]), float(g["t_lpf_s"])
        self.t, self.unit = self._unit_step()
        self.nadir_unit = float(self.unit.min())            # pu df per pu dp (negative)
        self.settle_unit = -1.0 / (self.D + 1.0 / self.mp)
        self.t_nadir = float(self.t[int(self.unit.argmin())])

    def _unit_step(self, t_end=4.0, h=1e-3):
        A = np.array([[-self.D / (2 * self.H), -1.0 / (2 * self.H)],
                      [1.0 / (self.mp * self.Tf), -1.0 / self.Tf]])
        B = np.array([-1.0 / (2 * self.H), 0.0])
        n = int(t_end / h)
        # exact discretisation
        M = np.zeros((3, 3)); M[:2, :2] = A; M[:2, 2] = B
        E = _expm(M * h)
        Ad, Bd = E[:2, :2], E[:2, 2]
        x = np.zeros(2); out = np.zeros(n + 1)
        for k in range(n):
            x = Ad @ x + Bd
            out[k + 1] = x[0]
        return np.arange(n + 1) * h, out

    # ------------------------------------------------------------------ API
    def event(self, dp_mw: float, s_eff_mva: float, df_pre_hz: float = 0.0):
        """Return (rocof0 [Hz/s], nadir_abs [Hz dev], df_settle_abs [Hz dev])."""
        if s_eff_mva <= 1e-9:
            return np.inf * np.sign(dp_mw or 1.0), -np.inf, -np.inf
        dp = dp_mw / s_eff_mva
        rocof0 = -dp * self.f0 / (2 * self.H)
        return (rocof0, df_pre_hz + dp * self.nadir_unit * self.f0,
                df_pre_hz + dp * self.settle_unit * self.f0)

    def max_step_mw(self, s_eff_mva: float, df_pre_hz: float, rocof_lim: float, df_lim: float):
        """Largest load step (MW) that keeps RoCoF and the nadir inside the limits."""
        by_rocof = rocof_lim * 2 * self.H / self.f0
        room = max(df_lim + min(df_pre_hz, 0.0), 0.0)      # less room if already under-frequency
        by_nadir = room / (abs(self.nadir_unit) * self.f0)
        return max(min(by_rocof, by_nadir), 0.0) * s_eff_mva

    def trajectory(self, dp_mw, s_eff_mva, df_pre_hz=0.0):
        return self.t, df_pre_hz + (dp_mw / s_eff_mva) * self.unit * self.f0


def _expm(M, terms=18):
    """Scaling-and-squaring Taylor matrix exponential (tiny matrices only)."""
    nrm = np.abs(M).sum(1).max()
    sq = max(0, int(np.ceil(np.log2(max(nrm, 1e-12)))) + 1)
    A = M / (2 ** sq)
    E = np.eye(M.shape[0]); term = np.eye(M.shape[0])
    for k in range(1, terms):
        term = term @ A / k
        E = E + term
    for _ in range(sq):
        E = E @ E
    return E
