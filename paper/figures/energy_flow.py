"""Energy-flow decomposition figure for the paper.

Reads the instrumented rollouts written by ``python -m sbx.energyflow run``
(``results/<tag>/<building>__<weather>.json``) and draws, for one scenario per
climate station:

  (a) the decomposition of the episode's energy+carbon loss into heating,
      cooling, boiler standby, fans and pump -- as fractions of that scenario's
      own loss, so the shape of the loss is comparable across climates;
  (b) the same decomposition in absolute reward units, so the cold climates'
      dominance is visible.

It also prints the markdown table the report quotes, and the ventilation-share
numbers the headline claim rests on. No simulator calls: every number is read
from a stored rollout.

Run:  starter-kit/.venv/bin/python paper/figures/energy_flow.py [--tag ef_dev30]
"""

from __future__ import annotations

import argparse
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

# One scenario per station, plus the two largest dev30 buildings. The order is
# cold to hot so the panels read left to right as a climate sweep.
SELECTION = [
    ("building_train_1/fairbanks_1", "Fairbanks"),
    ("building_train_2/fairbanks_2", "Fairbanks (large)"),
    ("building_train_7/denver_5", "Denver"),
    ("building_train_2/atlanta_3", "Atlanta"),
    ("building_train_8/phoenix_4", "Phoenix"),
    ("building_train_4/singapore_3", "Singapore"),
]

# Stacked-bar groups: components that share a mechanism share a colour.
GROUPS = [
    ("heat_vent", "heating: outside-air load", "#b2182b"),
    ("heat_rest", "heating: envelope/internal", "#ef8a62"),
    ("standby", "boiler standby/tank", "#fddbc7"),
    ("cool_vent", "cooling: outside-air load", "#2166ac"),
    ("cool_rest", "cooling: internal/solar/envelope", "#67a9cf"),
    ("fans", "fans", "#999999"),
    ("pump", "hot-water pump", "#4d4d4d"),
]


TAG = ["ef_shipped"]


def _components(record: dict) -> dict[str, float]:
    """Groups the stored per-component losses into the figure's categories."""
    comp = record["components"]
    out = {}
    for name, _, _ in GROUPS:
        if name == "standby":
            # Everything billed that the coils did not draw; may be negative
            # over a window where the tank discharged more than it was charged.
            value = comp["boiler_standby"]["loss_total"]
        elif name == "fans":
            value = comp["fan_supply"]["loss_total"] + comp["fan_exhaust"]["loss_total"]
        else:
            value = comp[name]["loss_total"]
        out[name] = -float(value)  # stored as a loss (negative); plot positive
    return out


def load(tag: str) -> dict[str, dict]:
    out = {}
    for scenario, _ in SELECTION:
        path = RESULTS / tag / (scenario.replace("/", "__") + ".json")
        if not path.exists():
            raise SystemExit(f"missing {path}; run: python -m sbx.energyflow run --tag {tag}")
        out[scenario] = json.loads(path.read_text())
    return out


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def table(records: dict[str, dict]) -> str:
    """Markdown: the reward-unit decomposition and the physical energy terms."""
    lines = [
        "| scenario | station | $\\bar T_\\mathrm{out}$ (°C) | comfort | energy | carbon "
        "| total | heating: outside-air | heating: envelope/internal | boiler standby "
        "| cooling: outside-air | cooling: internal/solar | fans | pump | residual |",
        "|" + "---|" * 15,
    ]
    for scenario, label in SELECTION:
        r = records[scenario]
        comp = _components(r)  # positive, grouped
        total = sum(comp.values())
        parts = sum(-v["loss_total"] for v in r["components"].values())
        lines.append(
            "| {scen} | {station} | {tout:.1f} | {comfort:.3f} | {energy:.2f} | {carbon:.2f} "
            "| {total:.2f} | {hv:.2f} | {hr:.2f} | {sb:.2f} | {cv:.2f} | {cr:.2f} "
            "| {fan:.3f} | {pump:.3f} | {resid:.3f} |".format(
                scen=scenario.split("/")[-1],
                station=label,
                tout=r["outside_temp_mean"] - 273.15,
                comfort=-r["loss_comfort"],
                energy=-r["loss_energy"],
                carbon=-r["loss_carbon"],
                total=-(r["loss_energy"] + r["loss_carbon"]),
                hv=comp["heat_vent"],
                hr=comp["heat_rest"],
                sb=comp["standby"],
                cv=comp["cool_vent"],
                cr=comp["cool_rest"],
                fan=comp["fans"],
                pump=comp["pump"],
                resid=total - parts,
            )
        )

    lines += [
        "",
        "| scenario | station | coil heat (kWh$_{th}$) | ventilation-carried (kWh$_{th}$, %) "
        "| marginal coil relief, 100% recirc (%) | chiller (kWh$_e$) | vent share of chiller (%) "
        "| supply fan (kWh$_e$) | exhaust fan (kWh$_e$) | pump (kWh$_e$) |",
        "|" + "---|" * 10,
    ]
    for scenario, label in SELECTION:
        r = records[scenario]
        fuel = r["fuel_totals"]
        stats = step_stats(TAG[0], scenario)
        coil = fuel["gas_coil_sum"] / 3.6e6
        vent_heat = stats.get("vent_heat_kwh", -fuel["vent_analytic_sum"] / 3.6e6)
        reheat = fuel["reheat_sum"] / 3.6e6
        reheat_cf = fuel["reheat_cf_sum"] / 3.6e6
        chiller = fuel["chiller_electrical_sum"] / 3.6e6
        vent_cool = fuel["chiller_vent_sum"] / 3.6e6
        vcp = f"{100.0 * vent_cool / chiller:.0f}%" if chiller > 1e-9 else "--"
        vhp = (
            f"{100.0 * vent_heat / coil:.0f}%"
            if coil > 1.0 and vent_heat <= coil
            else "--"
        )
        marg = (
            f"{100.0 * (reheat - reheat_cf) / reheat:.1f}%"
            if reheat > 1e-9
            else "--"
        )
        lines.append(
            "| {scen} | {station} | {coil:.1f} | {vh:.1f} ({vhp}) | {marg} "
            "| {chiller:.2f} | {vc:.2f} ({vcp}) | {fs:.2f} | {fe:.2f} | {pump:.2f} |".format(
                scen=scenario.split("/")[-1],
                station=label,
                coil=coil,
                vh=vent_heat,
                vhp=vhp,
                marg=marg,
                chiller=chiller,
                vc=vent_cool,
                vcp=vcp,
                fs=fuel["fan_supply_sum"] / 3.6e6,
                fe=fuel["fan_exhaust_sum"] / 3.6e6,
                pump=fuel["pump_sum"] / 3.6e6,
            )
        )
    return "\n".join(lines)


def step_stats(tag: str, scenario: str) -> dict[str, float]:
    """Step-level statistics that need the per-step series, not the totals."""
    path = RESULTS / tag / (scenario.replace("/", "__") + ".npz")
    if not path.exists():
        return {}
    with np.load(path) as npz:
        vent = np.maximum(0.0, -npz["q_vent_analytic"].astype(np.float64))
        coil = np.maximum(0.0, npz["gas_coil"].astype(np.float64))
        chiller = np.abs(npz["p_chiller"].astype(np.float64))
        recirc = npz["ahu"][:, :, 1].astype(np.float64)
        out = {}
        mask = coil > 0.0
        if mask.any():
            ratio = vent[mask] / coil[mask]
            out["heating_steps"] = int(mask.sum())
            out["vent_exceeds_coil_share"] = float((ratio > 1.0).mean())
            out["vent_to_coil_median"] = float(np.median(ratio))
        out["vent_heat_kwh"] = float(np.maximum(0.0, -npz["q_vent_analytic"]).sum() / 3.6e6)
        out["vent_cool_kwh"] = float(np.maximum(0.0, npz["q_vent_analytic"]).sum() / 3.6e6)
        if "coil_capacity" in npz:
            cap = npz["coil_capacity"].astype(np.float64)
            ok = (cap > 0.0) & mask
            if ok.any():
                out["coil_saturation_median"] = float(np.median(coil[ok] / cap[ok]))
        out["recirc_floored_share"] = float((recirc >= 0.6995).mean())
        out["recirc_mean"] = float(recirc.mean())
        cath = chiller > 0.0
        if cath.any():
            cv = npz["p_chiller_vent"].astype(np.float64)[cath]
            out["cooling_steps"] = int(cath.sum())
            out["cooling_vent_share_mean"] = float(np.clip(cv / chiller[cath], 0, 1).mean())
        return out


def claims(records: dict[str, dict]) -> str:
    """The report's headline numbers, computed rather than typed."""
    lines = []

    def fuel(r):
        return r["fuel_totals"]

    cold = [s for s, lab in SELECTION if "airbanks" in s or lab == "Denver"]
    tot = {}
    for scenario, _ in SELECTION:
        r = records[scenario]
        c = _components(r)
        tot[scenario] = sum(c.values())

    grand = sum(tot.values())
    hv = sum(_components(records[s])["heat_vent"] for s, _ in SELECTION)
    hr = sum(_components(records[s])["heat_rest"] for s, _ in SELECTION)
    st = sum(_components(records[s])["standby"] for s, _ in SELECTION)
    cv = sum(_components(records[s])["cool_vent"] for s, _ in SELECTION)
    cr = sum(_components(records[s])["cool_rest"] for s, _ in SELECTION)
    fn = sum(_components(records[s])["fans"] for s, _ in SELECTION)
    pu = sum(_components(records[s])["pump"] for s, _ in SELECTION)
    lines.append(
        "Across the six scenarios the energy+carbon loss is "
        f"{grand:.1f} reward units: heating {hv + hr + st:.1f} "
        f"({100 * (hv + hr + st) / grand:.0f}%), of which the marginal ventilation "
        f"duty is {hv:.1f} ({100 * hv / grand:.0f}% of the total, "
        f"{100 * hv / (hv + hr + st):.0f}% of the heating); cooling {cv + cr:.1f} "
        f"({100 * (cv + cr) / grand:.0f}%), of which ventilation {cv:.1f} "
        f"({100 * cv / (cv + cr):.0f}% of the cooling); fans {fn:.1f} "
        f"({100 * fn / grand:.0f}%); pump {pu:.2f} ({100 * pu / grand:.1f}%)."
    )
    for scenario, label in SELECTION:
        r = records[scenario]
        f = fuel(r)
        stats = step_stats(TAG[0], scenario)
        coil = f["gas_coil_sum"] / 3.6e6
        vent_heat = stats.get("vent_heat_kwh", -f["vent_analytic_sum"] / 3.6e6)
        reheat = f["reheat_sum"] / 3.6e6
        reheat_cf = f["reheat_cf_sum"] / 3.6e6
        outside = 100 * (
            1 - np.mean([a["recirculation_mean"] for a in r["ahu"].values()])
        )
        if coil > 1.0 and vent_heat <= coil:
            share = (
                f"{100 * vent_heat / coil:.0f}% of the {coil:.1f} kWh_th the reheat "
                f"coils delivered"
            )
        elif coil > 1.0:
            share = (
                f"more than the {coil:.1f} kWh_th the reheat coils delivered "
                f"({100 * vent_heat / max(coil, 1e-9):.0f}%), i.e. the coils cannot carry it"
            )
        else:
            share = f"against {coil:.1f} kWh_th of reheat, because almost no heating ran"
        lines.append(
            f"{label}: the outside-air stream at {outside:.0f}% of supply airflow "
            f"carries {vent_heat:.1f} kWh_th into the building over the episode, {share}"
            + (
                f"; recirculating everything would cut the coil duty by "
                f"{100 * (reheat - reheat_cf) / reheat:.1f}% only, because the coils are "
                f"water-limited (median duty {100 * stats.get('coil_saturation_median', float('nan')):.0f}% "
                f"of the air side's capacity)"
                if reheat > 1e-9 and stats.get("coil_saturation_median") is not None
                else ""
            )
            + "."
        )
    for scenario, label in SELECTION:
        f = fuel(records[scenario])
        chiller = f["chiller_electrical_sum"] / 3.6e6
        if chiller < 1e-6:
            continue
        lines.append(
            f"{label}: the chiller draws {chiller:.2f} kWh_e, of which "
            f"{100 * (f['chiller_vent_sum'] / 3.6e6) / chiller:.0f}% is attributable to the "
            f"outside air (the rest is internal, solar and envelope gain); the exhaust "
            f"fan, which exists only to make room for that outside air, adds "
            f"{f['fan_exhaust_sum'] / 3.6e6:.2f} kWh_e against "
            f"{f['fan_supply_sum'] / 3.6e6:.2f} kWh_e for the supply fan."
        )
    return "\n".join(lines)


def figure(records: dict[str, dict], name: str = "energy_flow") -> Path:
    labels = [label for _, label in SELECTION]
    grid = np.array(
        [[_components(records[scenario])[name] for name, _, _ in GROUPS]
         for scenario, _ in SELECTION]
    )
    totals = grid.sum(axis=1)
    shares = grid / np.where(totals[:, None] > 0, totals[:, None], 1.0)

    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.0, 1.0]}
    )
    for ax, data, xlabel, title in (
        (axes[0], shares, "share of the energy+carbon loss", "(a) composition"),
        (axes[1], grid, "energy+carbon loss (reward units)", "(b) absolute"),
    ):
        bottom = np.zeros(len(SELECTION))
        for column_index, (_, label, color) in enumerate(GROUPS):
            values = data[:, column_index]
            ax.barh(
                np.arange(len(SELECTION)),
                values,
                left=bottom,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                label=label,
            )
            bottom += values
        ax.set_yticks(np.arange(len(SELECTION)))
        ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        ax.invert_yaxis()
        ax.grid(axis="x", color="0.9", linewidth=0.5)
        ax.set_axisbelow(True)
    # Label the two ventilation segments inside panel (a): they are the claim.
    column = {group: i for i, (group, _, _) in enumerate(GROUPS)}
    for row in range(len(SELECTION)):
        for group in ("heat_vent", "cool_vent"):
            share = shares[row, column[group]]
            if share < 0.03:
                continue
            left = sum(shares[row, j] for j in range(column[group]))
            axes[0].annotate(
                f"{100 * share:.0f}%",
                (left + share / 2, row),
                ha="center",
                va="center",
                fontsize=6,
                color="white",
            )
    for row, total in enumerate(totals):
        axes[1].annotate(
            f"{-total:.1f}", (total, row), xytext=(3, 0), textcoords="offset points",
            va="center", fontsize=6.5,
        )
    axes[1].set_xlim(0, grid.sum(axis=1).max() * 1.16)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[::-1],
        names[::-1],
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    path = OUT / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="ef_shipped")
    parser.add_argument("--markdown", default=None, help="write the report body here")
    args = parser.parse_args(argv)
    TAG[0] = args.tag
    records = load(args.tag)
    print(table(records))
    print()
    for scenario, label in SELECTION:
        r = records[scenario]
        fuel = r["fuel_totals"]
        rec = r["reconciliation"]
        print(
            f"{label:>20}  steps={r['steps']:5d}  rec.err={rec.get('rel_error_vs_reward_terms', float('nan')):.1e}"
            f"  recirc={np.mean([a['recirculation_mean'] for a in r['ahu'].values()]):.3f}"
            f"  coil={fuel['gas_coil_sum'] / 3.6e6:7.1f}kWh"
            f"  vent-heat={-fuel['vent_analytic_sum'] / 3.6e6:7.1f}kWh"
            f"  chiller={fuel['chiller_electrical_sum'] / 3.6e6:6.2f}kWh_e"
            f"  standby={fuel['gas_standby_sum'] / 3.6e6:6.1f}kWh"
        )
        stats = step_stats(args.tag, scenario)
        if stats:
            print(
                f"{'':>20}  recirc mean {stats['recirc_mean']:.3f} "
                f"(at the 0.7 floor {100 * stats['recirc_floored_share']:.0f}% of steps)"
                + (
                    f", vent load > coil duty on {100 * stats['vent_exceeds_coil_share']:.0f}%"
                    f" of the {stats['heating_steps']} heating steps"
                    f" (median {stats['vent_to_coil_median']:.2f}x)"
                    if "vent_exceeds_coil_share" in stats
                    else ""
                )
                + (
                    f", ventilation {100 * stats['cooling_vent_share_mean']:.0f}% of the"
                    f" chiller draw (mean of per-step shares, "
                    f"{stats['cooling_steps']} cooling steps)"
                    if "cooling_vent_share_mean" in stats
                    else ""
                )
            )
    print()
    print(claims(records))
    path = figure(records)
    print(f"\nwrote {path}")
    if args.markdown:
        rows = []
        for scenario, label in SELECTION:
            r = records[scenario]
            stats = step_stats(args.tag, scenario)
            rows.append(
                f"| {label} | {scenario} | {r['steps']} | {r['num_zones']} "
                f"| {r['num_ahus']} | {r['mean_per_step_reward']:.6f} "
                f"| {stats.get('recirc_mean', float('nan')):.3f} "
                f"| {stats.get('coil_saturation_median', float('nan')):.2f} "
                f"| {r['reconciliation'].get('rel_error_energy', float('nan')):.1e} "
                f"| {r['reconciliation'].get('rel_error_carbon', float('nan')):.1e} |"
            )
        body = "\n".join(
            [
                "## Decomposition of the energy+carbon loss (reward units, positive)",
                "",
                table(records),
                "",
                "## Rollout metadata and the reconciliation check",
                "",
                "| station | scenario | steps | zones | AHUs | mean reward/step "
                "| recirculation | coil saturation | energy rel. err | carbon rel. err |",
                "|" + "---|" * 10,
                *rows,
                "",
                "## Key numbers",
                "",
                *[f"- {line}" for line in claims(records).splitlines()],
            ]
        )
        Path(args.markdown).write_text(body + "\n")
        print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
