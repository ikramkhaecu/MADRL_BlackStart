"""Small-sample statistics for RL comparisons (Agarwal et al., NeurIPS 2021)."""
from __future__ import annotations

import numpy as np


def iqm(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n < 4:
        return float(x.mean()) if n else float("nan")
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean())


def bootstrap_ci(x, stat=np.mean, n_boot=2000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = np.array([stat(rng.choice(x, len(x), replace=True)) for _ in range(n_boot)])
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def summarise(x):
    """[mean, std, ci_lo, ci_hi, iqm, n] over seeds."""
    x = np.asarray(x, float)
    lo, hi = bootstrap_ci(x)
    return [float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0, lo, hi, iqm(x), int(len(x))]
