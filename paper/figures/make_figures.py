"""Generate the paper's figures from the evaluation JSON files.

Run:  /opt/homebrew/Caskroom/miniconda/base/bin/python paper/figures/make_figures.py
Reads  <repo>/results/<tag>/building_*.json ; writes paper/figures/fig{1,2,3}_*.pdf
No simulator calls: every number below is read from a stored rollout.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    }
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # <repo>
RESULTS = ROOT / "results"
OUT = HERE

# ---------------------------------------------------------------- data access

def load(tag: str) -> list[dict]:
    files = sorted(RESULTS.glob(f"{tag}/building_*.json"))
    if not files:
        raise SystemExit(f"no result files for tag {tag!r} under {RESULTS}")
    return [json.loads(f.read_text()) for f in files]


def score(tag: str) -> float:
    """Benchmark score: mean over scenarios of |mean per-step reward|."""
    rows = load(tag)
    return float(np.mean([abs(r["mean_per_step_reward"]) for r in rows]))


def components(tag: str) -> tuple[float, float, float]:
    """Mean per-step comfort / energy / carbon loss (positive = cost, x1e-3)."""
    rows = load(tag)
    c = np.array(
        [
            [
                -r["components"]["sum_comfort"] / r["steps"],
                -r["components"]["sum_energy"] / r["steps"],
                -r["components"]["sum_carbon"] / r["steps"],
            ]
            for r in rows
        ]
    )
    return tuple(c.mean(axis=0))


def by_hour(tag: str) -> np.ndarray:
    """(3, 24) mean per-step comfort/energy/carbon loss by local hour."""
    rows = load(tag)
    acc = np.zeros((3, 24))
    for r in rows:
        bh = r["by_hour"]
        steps = np.asarray(bh["steps"], dtype=float)
        acc[0] += -np.asarray(bh["sum_comfort"]) / steps
        acc[1] += -np.asarray(bh["sum_energy"]) / steps
        acc[2] += -np.asarray(bh["sum_carbon"]) / steps
    return acc / len(rows)


def per_climate(tag: str) -> dict[str, tuple[float, float, float, float]]:
    """climate -> (n, mean total, mean energy, mean carbon) per-step loss."""
    out: dict[str, list[float]] = {}
    for r in load(tag):
        w = r["weather"].split("_")[0]
        n = r["steps"]
        a = out.setdefault(w, [0.0, 0.0, 0.0, 0.0])
        a[0] += 1
        a[1] += abs(r["mean_per_step_reward"])
        a[2] += -r["components"]["sum_energy"] / n
        a[3] += -r["components"]["sum_carbon"] / n
    return {w: (v[0], v[1] / v[0], v[2] / v[0], v[3] / v[0]) for w, v in out.items()}


def hour_comfort(tag: str, hour: int) -> float:
    """Mean per-step comfort loss inside one local hour, x1e-4."""
    rows = load(tag)
    vals = [-r["by_hour"]["sum_comfort"][hour] / r["by_hour"]["steps"][hour] for r in rows]
    return float(np.mean(vals)) * 1e4


# ---------------------------------------------------------------- figure 1

def fig1() -> None:
    series = [
        ("v1_dev30_shipped", "ctrl_v1 (first controller)"),
        ("refwu_dev30_shipped", "Wu et al. (base)"),
        ("ctrw_all", r"CtrlW (ours)"),
    ]
    labels = ["comfort ($0.6\\,C_t$)", "energy ($0.2\\,\\hat E_t$)", "carbon ($0.2\\,\\hat C_t$)"]
    colors = ["#c0392b", "#2c6fbb", "#4f8a3d"]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.0), sharey=True)
    handles = None
    for ax, (tag, name) in zip(axes, series):
        h = by_hour(tag) * 1e4
        hours = np.arange(24)
        layers = ax.stackplot(hours, h[0], h[1], h[2], colors=colors, labels=labels, linewidth=0)
        handles = handles or layers
        ax.set_xlim(0, 23)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_title(name)
        ax.set_xlabel("local hour")
        ax.axvspan(5.6, 18.4, color="0.85", alpha=0.25, lw=0, zorder=0)
    axes[0].set_ylabel(r"mean per-step loss ($\times10^{-4}$)")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.4,
    )
    fig.tight_layout(w_pad=0.5)
    fig.savefig(OUT / "fig1_hourly_loss.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2

def fig2() -> None:
    climates = ["fairbanks", "denver", "atlanta", "phoenix", "singapore"]
    methods = [
        ("pid_water_dev30_shipped", "pid_water (org.)", "#9aa0a6"),
        ("refzt_dev30_shipped", "zt-best", "#7f8fa6"),
        ("refwu_dev30_shipped", "Wu et al.", "#3b6ea5"),
        ("ctrw_all", "CtrlW", "#1b3f6b"),
    ]
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(6.6, 2.15), gridspec_kw={"width_ratios": [1.45, 1.0]}
    )
    x = np.arange(len(climates))
    w = 0.2
    for i, (tag, name, col) in enumerate(methods):
        pc = per_climate(tag)
        vals = [pc[c][1] * 1e3 for c in climates]
        ax.bar(x + (i - 1.5) * w, vals, w, label=name, color=col, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(climates)
    ax.set_ylabel(r"dev30 total loss ($\times10^{-3}$)")
    ax.set_ylim(0, 108)
    ax.legend(frameon=False, ncol=2, loc="upper right", handlelength=1.1, columnspacing=0.8)
    ax.set_title("total loss by climate")

    pc = per_climate("ctrw_all")
    comp = np.array([[pc[c][2], pc[c][3]] for c in climates]) * 1e3
    ax2.bar(x, comp[:, 0], 0.6, label="energy", color="#2c6fbb", linewidth=0)
    ax2.bar(x, comp[:, 1], 0.6, bottom=comp[:, 0], label="carbon", color="#4f8a3d", linewidth=0)
    ax2.set_xticks(x)
    ax2.set_xticklabels([c[:4] for c in climates])
    ax2.set_ylabel(r"CtrlW loss ($\times10^{-3}$)")
    ax2.legend(frameon=False, loc="upper right", handlelength=1.1)
    ax2.set_title("CtrlW: the residual is the bill")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "fig2_climate_loss.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3

def fig3() -> None:
    # Wu's base preheats at most 1 h: its ramp lives inside the 60-min comfort_mode_soon
    # flag, so it is the 1 h point of the sweep.
    pts = [
        (1.0, "refwu_dev30_shipped"),
        (2.0, "ctrw_d2"),
        (3.0, "ctrw_d3"),
        (4.0, "ctrw_d4"),
    ]
    hs = [p[0] for p in pts]
    comf = [hour_comfort(t, 6) for _, t in pts]
    tot = [score(t) * 1e3 for _, t in pts]

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(6.6, 1.95), gridspec_kw={"width_ratios": [1.0, 1.35]}
    )
    ax.bar(hs, comf, 0.55, color="#c0392b", linewidth=0, label="hour-06 comfort")
    ax.set_xlabel("preheat horizon (h)")
    ax.set_ylabel(r"hour-06 comfort loss ($\times10^{-4}$)", color="#c0392b")
    ax.set_xticks(hs)
    ax.set_ylim(0, max(comf) * 1.30)
    ax.tick_params(axis="y", colors="#c0392b")

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(hs, tot, "o-", color="#1b3f6b", lw=1.2, ms=3.5, label="dev30 score")
    ax2.set_ylabel(r"dev30 score ($\times10^{-3}$)", color="#1b3f6b")
    ax2.tick_params(axis="y", colors="#1b3f6b")
    ax2.set_ylim(7.30, 7.62)
    ax2.axvline(2.18, color="0.6", ls=":", lw=0.8)
    ax.set_title("readiness saturates, preheat energy does not")

    # (b) where the morning loss moves
    hours = np.arange(3, 10)
    for tag, name, col, ls in [
        ("refwu_dev30_shipped", "Wu et al. (1 h)", "#3b6ea5", "-"),
        ("ctrw_d2", "CtrlW, 2 h preheat", "#c0392b", "--"),
    ]:
        rows = load(tag)
        prof = np.array(
            [
                [-r["by_hour"]["sum_reward"][h] / r["by_hour"]["steps"][h] for h in hours]
                for r in rows
            ]
        ).mean(axis=0)
        axb.plot(hours, prof * 1e4, ls, marker="o", ms=3, lw=1.2, color=col, label=name)
    axb.set_xticks(hours)
    axb.set_xlabel("local hour")
    axb.set_ylabel(r"total loss ($\times10^{-4}$)")
    axb.set_title("the readiness constraint caps the morning peak")
    axb.legend(frameon=False, loc="upper right", handlelength=1.6)
    axb.axvline(6, color="0.75", lw=0.6, zorder=0)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / "fig3_preheat.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("wrote", sorted(p.name for p in OUT.glob("fig*.pdf")))
    print("scores", {t: round(score(t), 7) for t in
                     ["refwu_dev30_shipped", "ctrw_d2", "ctrw_d3", "ctrw_d4", "ctrw_all", "ctrw_all_corr"]})
    print("hour06 comfort", {t: round(hour_comfort(t, 6), 2) for t in
                             ["refwu_dev30_shipped", "ctrw_d2", "ctrw_d3", "ctrw_d4", "ctrw_all"]})
    print("CtrlW spectra", [round(x, 5) for x in components("ctrw_all")])
