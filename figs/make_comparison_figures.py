"""Generate all results figures from the JSON files written by ``sim.run_experiments``.

  fig5_training_curves_<sys>.png  learning curves, mean +/- std over seeds (UNSHAPED return)
  fig9_algorithm_benchmark.png    held-out test bars, 14-bus
  fig10_metric_comparison.png     six-metric comparison on the largest system that was run
  fig11_constraint_analysis.png   component ablation
  fig12_scalability.png           priority-load stress test
  fig13_scenarios.png             S1-S5
  fig14_restoration_trace.png     electrical time series of one restoration (f, V, P, SoC, LRR)

Usage: python figs/make_comparison_figures.py [--results sim/results] [--out figs]
"""
import argparse
import json
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, GRID, BASE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
mpl.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": BASE,
    "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False})
HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = {"ppo": "#2a78d6", "mappo": "#eb6834", "dqn": "#1baf7a", "iql": "#eda100", "qmix": "#e87ba4",
          "vdn": "#008300", "pi-mappo": "#4a3aa7"}
LABEL = {"mappo": "MAPPO", "ppo": "PPO", "dqn": "DQN", "iql": "IQL", "qmix": "QMIX", "vdn": "VDN",
         "pi-mappo": "PI-MAPPO", "mappo+mask": "MAPPO+mask", "mappo+shape": "MAPPO+PBRS",
         "mappo+feat": "MAPPO+$\\phi$", "pi-mappo-nomask": "PI $-$mask", "pi-mappo-nofeat": "PI $-\\phi$",
         "pi-mappo-noshape": "PI $-$PBRS", "pi-mappo-nophys": "PI $-\\mathcal{L}^{PI}$", "pi-mappo-nodan": "PI $-$DAN"}
LIMIT = "#d03b3b"
SYSNAME = {"14": "IEEE 14-bus", "39": "IEEE 39-bus", "118": "IEEE 118-bus"}


def _col(a):
    return COLORS.get(a.replace("+pi", ""), "#777777")


def _deframe(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def _smooth(x, win):
    x = np.asarray(x, float)
    if len(x) < win:
        return x
    k = np.ones(win) / win
    return np.convolve(np.pad(x, (win - 1, 0), mode="edge"), k, mode="valid")


def _load(R, fn):
    p = os.path.join(R, fn)
    return json.load(open(p)) if os.path.exists(p) else None


def _stamp(fig, meta):
    if meta and meta.get("label"):
        fig.text(0.995, 0.005, f"{meta['label']}: {len(meta['seeds'])} seeds x {meta['episodes']} episodes",
                 ha="right", va="bottom", fontsize=7, color=LIMIT)


def fig_training_curves(b, s, out, meta):
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.4))
    keys = [("ret", "Unshaped episode return"), ("lrr", "Load recovery ratio"),
            ("fvr", "Frequency violation rate"), ("failed", "Collapsed episodes (share)")]
    for ax, (k, ttl) in zip(axs, keys):
        for a, d in b[s].items():
            m, sd = np.array(d["hist"][k]["mean"]), np.array(d["hist"][k]["std"])
            win = max(len(m) // 30, 5)
            ms_, ss = _smooth(m, win), _smooth(sd, win)
            x = np.arange(1, len(m) + 1)
            ax.plot(x, ms_, color=_col(a), lw=2.2 if a == "pi-mappo" else 1.3, label=LABEL.get(a, a))
            ax.fill_between(x, ms_ - ss, ms_ + ss, color=_col(a), alpha=0.12, lw=0)
        ax.set_title(ttl); ax.set_xlabel("Training episode"); _deframe(ax)
    axs[0].legend(ncol=2)
    fig.suptitle(f"{SYSNAME[s]}: training curves (mean $\\pm$ std over seeds, exploration policy)", y=1.03)
    _stamp(fig, meta)
    fig.savefig(os.path.join(out, f"fig5_training_curves_{s}.png")); plt.close(fig)


def _bars(ax, d, key, title, scale=1.0, limit=None):
    names = list(d)
    mean = [d[a]["metrics"][key][0] * scale for a in names]
    std = [d[a]["metrics"][key][1] * scale for a in names]
    ax.bar(range(len(names)), mean, yerr=std, color=[_col(a) for a in names], capsize=3,
           error_kw=dict(lw=0.8, ecolor=INK2))
    ax.set_xticks(range(len(names))); ax.set_xticklabels([LABEL.get(a, a) for a in names], rotation=40, ha="right")
    ax.set_title(title); _deframe(ax)
    if limit is not None:
        ax.axhline(limit, color=LIMIT, ls="--", lw=1)


def fig_bars(b, s, out, meta, fn, title):
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (k, ttl, sc, lim) in zip(axs.ravel(), [
            ("ret", "Unshaped return", 1, None), ("lrr", "Load recovery ratio", 1, None),
            ("buses", "Energised nodes", 1, None), ("vvr", "Voltage violation rate (%)", 100, None),
            ("fvr", "Frequency violation rate (%)", 100, None), ("failed", "Collapsed episodes (%)", 100, None)]):
        _bars(ax, b[s], k, ttl, sc, lim)
    fig.suptitle(title, y=1.0); fig.tight_layout(); _stamp(fig, meta)
    fig.savefig(os.path.join(out, fn)); plt.close(fig)


def fig_scalability(sc, out, meta):
    lv = [str(x) for x in sc["levels"]]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (k, ttl, s_) in zip(axs, [("lrr", "Load recovery ratio", 1), ("ret", "Unshaped return", 1),
                                      ("fvr", "Frequency violation rate (%)", 100)]):
        m = [sc["metrics"][x][k][0] * s_ for x in lv]; e = [sc["metrics"][x][k][1] * s_ for x in lv]
        ax.errorbar(sc["levels"], m, yerr=e, color=COLORS["pi-mappo"], marker="o", capsize=3)
        ax.set_xlabel("Loads with 1.5x priority demand"); ax.set_title(ttl); _deframe(ax)
    fig.suptitle(f"PI-MAPPO stress test, {SYSNAME[sc['_system']]} (no retraining)", y=1.03); _stamp(fig, meta)
    fig.savefig(os.path.join(out, "fig12_scalability.png")); plt.close(fig)


def fig_scenarios(b, s, out, meta):
    algos = [a for a in b[s] if "scenarios" in b[s][a]]
    if not algos:
        return
    scs = list(b[s][algos[0]]["scenarios"])
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    w = 0.8 / len(algos)
    for ax, (k, ttl, sc_) in zip(axs, [("lrr", "Load recovery ratio", 1), ("fvr", "Frequency violation rate (%)", 100),
                                       ("failed", "Collapsed episodes (%)", 100)]):
        for j, a in enumerate(algos):
            m = [b[s][a]["scenarios"][x][k][0] * sc_ for x in scs]
            ax.bar(np.arange(len(scs)) + j * w, m, w, color=_col(a), label=LABEL.get(a, a))
        ax.set_xticks(np.arange(len(scs)) + 0.4 - w / 2); ax.set_xticklabels(scs); ax.set_title(ttl); _deframe(ax)
    axs[0].legend(ncol=2, fontsize=7)
    fig.suptitle(f"Scenario evaluation S1-S5, {SYSNAME[s]}", y=1.03); _stamp(fig, meta)
    fig.savefig(os.path.join(out, "fig13_scenarios.png")); plt.close(fig)


def fig_trace(b, s, out, meta, lim_f=0.5, band=(0.95, 1.05)):
    tr = b[s].get("pi-mappo", {}).get("trace")
    if not tr:
        return
    t = [x["t"] for x in tr]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.4))
    axs[0].step(t, [x["f_signed"] for x in tr], where="post", color=COLORS["pi-mappo"], label="settling $\\Delta f$")
    axs[0].plot(t, [-x["nadir"] for x in tr], "v", color=COLORS["mappo"], ms=4, label="$-|$nadir$|$ of the step's event")
    for y in (-lim_f, lim_f):
        axs[0].axhline(y, color=LIMIT, ls="--", lw=1)
    axs[0].set_title("Frequency deviation (Hz)"); axs[0].legend(fontsize=7)
    axs[1].fill_between(t, [x["vmin"] for x in tr], [x["vmax"] for x in tr], color=COLORS["dqn"], alpha=0.35, step="post")
    for y in band:
        axs[1].axhline(y, color=LIMIT, ls="--", lw=1)
    axs[1].set_title("Bus-voltage envelope (pu)")
    P = np.array([x["p_unit"] for x in tr])
    for k in range(min(P.shape[1], 6)):
        axs[2].step(t, P[:, k], where="post", label=f"unit {k}")
    axs[2].set_title("GFM unit power (MW)"); axs[2].legend(fontsize=7, ncol=2)
    axs[3].step(t, [x["lrr"] for x in tr], where="post", color=COLORS["pi-mappo"], label="LRR")
    S = np.array([x["soc"] for x in tr]); ess = S.max(0) > 0
    if ess.any():
        axs[3].step(t, S[:, ess].mean(1), where="post", color=COLORS["iql"], label="mean SoC")
    axs[3].set_title("Load recovery and storage"); axs[3].legend(fontsize=7)
    for ax in axs:
        ax.set_xlabel("Step"); _deframe(ax)
    fig.suptitle(f"One held-out restoration by PI-MAPPO, {SYSNAME[s]}", y=1.03); _stamp(fig, meta)
    fig.savefig(os.path.join(out, f"fig14_restoration_trace_{s}.png")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "..", "sim", "results"))
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    R, out = a.results, a.out
    os.makedirs(out, exist_ok=True)
    meta, b = _load(R, "meta.json"), _load(R, "benchmark.json")
    if b:
        for s in b:
            fig_training_curves(b, s, out, meta); fig_trace(b, s, out, meta)
        s0, big = list(b)[0], list(b)[-1]
        fig_bars(b, s0, out, meta, "fig9_algorithm_benchmark.png", f"Held-out test performance, {SYSNAME[s0]}")
        fig_bars(b, big, out, meta, "fig10_metric_comparison.png", f"Held-out test performance, {SYSNAME[big]}")
        fig_scenarios(b, s0, out, meta)
    ab = _load(R, "ablation.json")
    if ab:
        s0 = list(ab)[0]
        fig_bars(ab, s0, out, meta, "fig11_constraint_analysis.png", f"Component ablation, {SYSNAME[s0]}")
    fair = _load(R, "fair.json")
    if fair:
        s0 = list(fair)[0]
        fig_bars(fair, s0, out, meta, "fig9b_fair_comparison.png", f"All algorithms behind the same PI mask, {SYSNAME[s0]}")
    sc = _load(R, "scalability.json")
    if sc:
        fig_scalability(sc, out, meta)
    print("figures written to", out)


if __name__ == "__main__":
    main()
