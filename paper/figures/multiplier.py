"""Figure: the ventilation multiplier of the heating bill, and where CtrlW5's
12.8 units come from.

(a) gas billed per unit of net zone heat, from the plant's own air-side law
    gas/Q = 1 + 0.3 (T_zone - T_out) / (T_dis - T_zcl)  at the mandatory 30 %
    outside-air floor and the 308.15 K discharge cap -- the line that makes a
    cold building cheaper to heat per delivered watt.  The two operating points
    are the ones the text quotes (2.00 at 294 K, 1.18 at 270 K, anchor ambient).
(b) the anchor's weekly loss (reward units) by hour block, CtrlW vs CtrlW5.

Every number in (b) is from docs/control_gap.md §2 ("By hour", each run's own
by_hour); (a) is the closed-form law with the anchor's mean outside temperature
(-26.3 C, i.e. 246.85 K) and one warmer ambient for scale.

Run:  python3 paper/figures/multiplier.py
"""

from __future__ import annotations

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
OUT = HERE / "multiplier.pdf"

T_DIS = 308.15  # K, the discharge cap (35 C)
KELVIN = 273.15


def multiplier(t_zone_k: float, t_out_k: float) -> float:
    """gas per unit of net zone heat; 30 % outside air, discharge at the cap."""
    return 1.0 + 0.3 * (t_zone_k - t_out_k) / (T_DIS - t_zone_k)


def main() -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.0, 1.5))

    # ---- (a) the law -----------------------------------------------------
    tz_c = np.linspace(-8.0, 24.0, 400)
    tz_k = tz_c + KELVIN
    for t_out_c, style, label in [
        (-26.3, "-", "anchor, $-26.3\\,^\\circ$C"),
        (-10.0, "--", "$-10\\,^\\circ$C"),
    ]:
        ax.plot(tz_c, multiplier(tz_k, t_out_c + KELVIN), style, lw=1.3, label=label)

    for t_op_c, txt, dx, dy in [(20.85, "2.00", -5.2, 0.02), (-3.15, "1.18", 1.2, 0.10)]:
        ax.plot([t_op_c], [multiplier(t_op_c + KELVIN, 246.85)], "o", ms=3.0, color="k")
        ax.annotate(
            f"{txt}",
            xy=(t_op_c, multiplier(t_op_c + KELVIN, 246.85)),
            xytext=(t_op_c + dx, multiplier(t_op_c + KELVIN, 246.85) + dy),
            fontsize=7,
        )

    ax.set_xlabel("zone temperature ($^\\circ$C)")
    ax.set_ylabel("gas per unit net heat")
    ax.set_title("(a) the bill's ventilation multiplier", loc="left")
    ax.set_ylim(0.9, 2.6)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.80), handlelength=1.6)
    ax.text(
        0.03,
        0.86,
        "$\\mathrm{gas}/Q=1+\\frac{0.3\\,(T_z-T_\\mathrm{out})}{T_\\mathrm{dis}-T_z}$,"
        " $T_\\mathrm{dis}=308.15$ K",
        transform=ax.transAxes,
        fontsize=7,
    )

    # ---- (b) where the release's units come from -------------------------
    # src: docs/control_gap.md §2 "By hour" (results/cg_ctrlw_anchor, cg_w5_final by_hour)
    blocks = ["00\u201304", "04\u201306", "06\u201312", "12\u201319", "19\u201324"]
    ctrlw = [-3.365, -14.450, -41.626, -30.291, -3.390]
    ctrlw5 = [-0.030, -11.778, -40.968, -27.177, -0.351]

    x = np.arange(len(blocks))
    w = 0.38
    bx.bar(x - w / 2, np.abs(ctrlw), w, label="CtrlW", color="0.75", edgecolor="k", lw=0.4)
    bx.bar(x + w / 2, np.abs(ctrlw5), w, label="CtrlW5", color="0.25", edgecolor="k", lw=0.4)
    for i, (a, b) in enumerate(zip(ctrlw, ctrlw5)):
        gain = abs(a) - abs(b)
        bx.text(i, max(abs(a), abs(b)) + 0.8, f"{gain:+.1f}", ha="center", fontsize=6.5)
    bx.set_xticks(x)
    bx.set_xticklabels(blocks)
    bx.set_xlabel("local hour")
    bx.set_ylabel("anchor loss (reward units)")
    bx.set_title("(b) anchor week by hour block", loc="left")
    bx.set_ylim(0, 50)
    bx.legend(frameon=False, loc="upper left")

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
