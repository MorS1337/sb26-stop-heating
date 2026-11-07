"""Reward-anatomy figure for the paper.

  (a) the cold anchor (``building_train_1/fairbanks_1``, first 2 016 steps = the
      7 simulated days the public forum compares on): the energy-plus-carbon
      loss of every local hour, stacked by mechanism, so the 04--08 h capacity-
      limited morning and the envelope/outside-air split are both visible;
  (b) the same decomposition for the six full-length (8 641-step) scenarios of
      ``results/ef_shipped``, in absolute reward units, so the concentration of
      the whole bill in three scenarios is visible.

No simulator calls: (a) reads the per-step series stored by
``python -m sbx.energyflow run`` (``results/ef_shipped/*.npz``), (b) the
per-scenario aggregates (``results/ef_shipped/*.json``). Both are the shipped
simulator and policy ``policies.ctrl_w:CtrlW``.

Run:  starter-kit/.venv/bin/python paper/figures/anatomy.py
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
        "legend.fontsize": 6.5,
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
ROOT = HERE.parent.parent
RESULTS = ROOT / "results"
OUT = HERE

TAG = "ef_shipped"
ANCHOR = "building_train_1/fairbanks_1"
ANCHOR_STEPS = 2016  # the 7-day window used by the public forum comparison

# Cold to hot; the two largest dev30 buildings are kept separate.
SELECTION = [
    ("building_train_1/fairbanks_1", "Fairbanks"),
    ("building_train_2/fairbanks_2", "Fairbanks (large)"),
    ("building_train_7/denver_5", "Denver"),
    ("building_train_2/atlanta_3", "Atlanta"),
    ("building_train_8/phoenix_4", "Phoenix"),
    ("building_train_4/singapore_3", "Singapore"),
]

# (key, label, colour, panel-a?) -- panel (a) has no cooling at the anchor, so
# the three cooling groups are drawn only in (b).
GROUPS = [
    ("heat_rest", "heating: envelope/internal", "#ef8a62", True),
    ("heat_vent", "heating: outside-air load", "#b2182b", True),
    ("standby", "boiler standby/tank", "#fddbc7", True),
    ("cool_rest", "cooling: internal/solar/envelope", "#67a9cf", False),
    ("cool_vent", "cooling: outside-air load", "#2166ac", False),
    ("fans", "fans", "#999999", True),
    ("pump", "hot-water pump", "#4d4d4d", True),
]

SERIES = {
    "heat_rest": ["loss_heat_rest"],
    "heat_vent": ["loss_heat_vent"],
    "standby": ["loss_boiler_standby"],
    "cool_rest": ["loss_cool_rest"],
    "cool_vent": ["loss_cool_vent"],
    "fans": ["loss_fan_supply", "loss_fan_exhaust"],
    "pump": ["loss_pump"],
}


def anchor_hourly() -> np.ndarray:
    """(24, n_groups) loss per hour-of-day over the 7-day anchor, positive."""
    z = np.load(RESULTS / TAG / (ANCHOR.replace("/", "__") + ".npz"))
    hour = z["hour"][:ANCHOR_STEPS].astype(int)
    out = np.zeros((24, len(GROUPS)))
    for g, (name, _, _, _) in enumerate(GROUPS):
        series = sum(z[k][:ANCHOR_STEPS].astype(np.float64) for k in SERIES[name])
        for h in range(24):
            out[h, g] = -series[hour == h].sum()
    return out


def scenario_totals() -> tuple[list[str], np.ndarray]:
    labels, rows = [], []
    for scenario, short in SELECTION:
        rec = json.loads((RESULTS / TAG / (scenario.replace("/", "__") + ".json")).read_text())
        comp = rec["components"]
        values = []
        for name, _, _, _ in GROUPS:
            if name == "standby":
                v = comp["boiler_standby"]["loss_total"]
            elif name == "fans":
                v = comp["fan_supply"]["loss_total"] + comp["fan_exhaust"]["loss_total"]
            else:
                v = comp[name]["loss_total"]
            values.append(-float(v))
        rows.append(values)
        labels.append(short)
    return labels, np.array(rows)


def main() -> None:
    hourly = anchor_hourly()
    labels, totals = scenario_totals()

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(5.5, 2.45),
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.30},
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.29)

    # ---------------------------------------------------------------- panel (a)
    x = np.arange(24)
    bottom = np.zeros(24)
    for g, (name, label, colour, in_a) in enumerate(GROUPS):
        if not in_a:
            continue
        ax_a.bar(x, hourly[:, g], bottom=bottom, width=0.78, color=colour,
                 edgecolor="none", label=label, zorder=3)
        bottom += hourly[:, g]
    ax_a.set_xlim(-0.7, 23.7)
    ax_a.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    ax_a.set_xlabel("local hour of day")
    ax_a.set_ylabel("energy+carbon loss\n(reward units, 7 days summed)")
    ax_a.set_title("(a) cold anchor, 7 days, by hour")
    ax_a.grid(axis="y", color="0.88", lw=0.5, zorder=0)

    morning = hourly[4:9].sum() / hourly.sum()
    ax_a.axvspan(3.6, 8.4, color="0.2", alpha=0.06, zorder=1, lw=0)
    ax_a.set_ylim(0, bottom.max() * 1.30)
    ax_a.text(
        9.2,
        bottom.max() * 1.28,
        f"04\u201308 h: {morning * 100:.1f}% of the bill\n"
        f"({hourly[4:9].sum():.1f} of {hourly.sum():.1f} units,\n21% of the hours)",
        fontsize=6.5,
        va="top",
        ha="left",
        zorder=5,
    )

    # ---------------------------------------------------------------- panel (b)
    y = np.arange(len(SELECTION))[::-1]
    left = np.zeros(len(SELECTION))
    for g, (name, label, colour, _) in enumerate(GROUPS):
        ax_b.barh(y, totals[:, g], left=left, height=0.72, color=colour,
                  edgecolor="none", zorder=3)
        left += totals[:, g]
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(labels)
    ax_b.set_xlabel("energy+carbon loss (reward units)")
    ax_b.set_title("(b) six scenarios, full episodes")
    ax_b.grid(axis="x", color="0.88", lw=0.5, zorder=0)
    for yy, total in zip(y, left):
        ax_b.text(total + 4, yy, f"{total:.0f}", va="center", fontsize=6.5)
    ax_b.set_xlim(0, left.max() * 1.18)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for _, _, colour, _ in GROUPS]
    names = [label for _, label, _, _ in GROUPS]
    fig.legend(handles, names, loc="lower center", ncol=4, frameon=False,
               handlelength=1.1, columnspacing=1.3, bbox_to_anchor=(0.5, -0.015))

    fig.savefig(OUT / "anatomy.pdf")
    print(f"wrote {OUT / 'anatomy.pdf'}")
    print(f"anchor hour split: 04-08 {morning:.3f}; total {hourly.sum():.3f}")
    print("scenario totals:", dict(zip(labels, np.round(left, 1))))


if __name__ == "__main__":
    main()
