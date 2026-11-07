"""Figure: what a frost floor costs.

Sweep of the night standby floor (CtrlW5's other two knobs held fixed), anchor
episode (building_train_1/fairbanks_1, 2016 steps).  Left axis: anchor total
reward; right axis: the coldest room temperature the episode reaches.  The two
reference lines are CtrlW's own 282 K floor, and the 0 C / 5 C frost criteria.

All points are measured; the source table is docs/control_gap.md §8.2, with
the 274/276/280 K rows re-read from the frost recorder JSONs in
scratch/ControlGap/ (same totals to three decimals).  The minimum room
temperature marked at 278 K is the number quoted in §8.1 (30-day episode, +4.66
C minimum, zero room-hours below 0 C).

Run:  python3 paper/figures/frost.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
OUT = HERE / "frost.pdf"

# src: docs/control_gap.md §8.2 (floor sweep, 2016 steps) and §8.1 (min T)
FLOORS_K = [262, 274, 276, 278, 280, 282]
ANCHOR_TOTAL = [-80.306, -83.090, -84.638, -86.681, -89.575, -93.127]
MIN_ROOM_C = [-8.58, 0.63, 2.66, 4.66, 6.53, 8.55]


def main() -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    bx = ax.twinx()

    ax.plot(FLOORS_K, ANCHOR_TOTAL, "-o", ms=3.0, lw=1.3, color="k")
    bx.plot(FLOORS_K, MIN_ROOM_C, "--s", ms=2.8, lw=1.0, color="0.45")

    ax.axhline(-93.127, color="0.75", lw=0.7, ls=":")
    ax.annotate("CtrlW, 282 K", xy=(281.6, -92.5), fontsize=6.3, color="0.35",
                ha="right", va="bottom")
    bx.axhline(0.0, color="0.85", lw=0.7)
    bx.axhline(5.0, color="0.85", lw=0.7, ls="--")
    bx.annotate("0 $^\\circ$C", xy=(259.3, 0.3), fontsize=6.3, color="0.35")
    bx.annotate("5 $^\\circ$C", xy=(259.3, 6.2), fontsize=6.3, color="0.35")

    ax.axvline(278, color="C0", lw=0.8, ls="-.")
    ax.annotate("shipped 278 K", xy=(278.5, -87.0), fontsize=6.3, color="C0",
                rotation=90, va="center")

    ax.annotate("anchor total", xy=(263.8, -79.9), fontsize=6.5)
    bx.annotate("coldest room", xy=(276.4, 7.6), fontsize=6.5, color="0.35")

    ax.set_xlabel("night standby floor $T_\\mathrm{sb}$ (K)")
    ax.set_ylabel("anchor total reward")
    bx.set_ylabel("coldest room ($^\\circ$C)")
    ax.set_xlim(258.5, 285.0)
    ax.set_ylim(-95.5, -78.5)
    bx.set_ylim(-11.5, 11.5)
    ax.set_xticks([262, 268, 274, 278, 282])

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
