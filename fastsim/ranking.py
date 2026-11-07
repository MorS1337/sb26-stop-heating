"""Policy-ranking agreement: does the ROM order policies the way the real sim does?

For each building, a fixed surrogate (basis from the strip's logged fields, maps
refitted by operator inference on that strip's exact transitions) replaces only
``FiniteDifferenceSolver.solve`` in the real environment. Five CtrlW parameter
variants are then run for 2880 steps with the real solve and with the ROM solve,
and the two reward orderings are compared with Spearman and Kendall.

This is the question that decides whether the surrogate is usable for tuning:
an absolute reward bias shifts every candidate equally and cancels in a ranking;
what would break a tuning run is a bias that depends on the policy.

Usage:
  starter-kit/.venv/bin/python -m fastsim.ranking --steps 2880 --m 200 \
      --out fastsim/results/ranking.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VARIANTS = {
    "default": {},
    "margin0.0": {"comfort_margin": 0.0},
    "margin0.4": {"comfort_margin": 0.4},
    "preheat1h": {"early_preheat_hours": 1.0},
    "standby285": {"standby_minimum": 285.0},
}

SCENARIOS = {
    "building_train_8": "building_train_8/fairbanks_8",
    "building_train_1": "building_train_1/atlanta_2",
}


def _corr(real: np.ndarray, rom: np.ndarray) -> dict:
    from scipy.stats import kendalltau, spearmanr  # noqa: PLC0415

    if len(real) < 3:
        return {}
    sp = spearmanr(real, rom)
    kd = kendalltau(real, rom)
    return dict(
        spearman=float(sp.statistic),
        spearman_p=float(sp.pvalue),
        kendall=float(kd.statistic),
        kendall_p=float(kd.pvalue),
        top1=bool(np.argmax(real) == np.argmax(rom)),
        bottom1=bool(np.argmin(real) == np.argmin(rom)),
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=2880)
    p.add_argument("--m", type=int, default=200)
    p.add_argument("--lam", type=float, default=1e-4)
    p.add_argument("--sim", default="shipped")
    p.add_argument("--buildings", default=",".join(SCENARIOS))
    p.add_argument("--fit", action="store_true",
                   help="use operator inference for the reduced maps (otherwise projection)")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    from . import closedloop, scen  # noqa: PLC0415
    from .opinfer import build as build_fitted  # noqa: PLC0415
    from .replay import load_rollout  # noqa: PLC0415
    from .rom import ROM, Ops  # noqa: PLC0415

    scen.pin_blas()
    scen.boot(a.sim)
    import importlib  # noqa: PLC0415

    policy_cls = importlib.import_module("policies.ctrl_w").CtrlW
    defaults = dict(policy_cls.PARAMS)

    rows = []
    for building in a.buildings.split(","):
        scenario = SCENARIOS[building]
        _, _, snaps = load_rollout(scenario)
        if a.fit:
            rom, o, maps = build_fitted(building, m=a.m, basis_scenario=scenario,
                                        fit=True, lam=a.lam)
        else:
            o = Ops(building)
            rom = ROM(building, m=a.m, kind="hybrid", snaps=snaps, ops=o)
            maps = None
        for name, overrides in VARIANTS.items():
            for mode in ("real", "rom"):
                policy_cls.PARAMS = {**defaults, **overrides}
                r = closedloop.run(
                    scenario, steps=a.steps, sim=a.sim, snaps=snaps,
                    params=overrides, rom_in=rom, ops_in=o,
                    use_rom=(mode == "rom"),
                )
                row = dict(
                    building=building, scenario=scenario, variant=name,
                    overrides=overrides, mode=mode,
                    total_reward=r["total_reward"],
                    mean_per_step=r["mean_per_step_reward"],
                    steps=r["steps"],
                )
                rows.append(row)
                print(f"{building:16s} {name:11s} {mode:4s} "
                      f"{r['total_reward']:+.6f}  (mean {r['mean_per_step_reward']:+.6f})")

    summary = {}
    for building in a.buildings.split(","):
        sel = [r for r in rows if r["building"] == building]
        real = np.array([next(r["total_reward"] for r in sel
                              if r["variant"] == v and r["mode"] == "real")
                         for v in VARIANTS])
        rms = np.array([next(r["total_reward"] for r in sel
                             if r["variant"] == v and r["mode"] == "rom")
                        for v in VARIANTS])
        summary[building] = dict(
            variants=list(VARIANTS), real=real.tolist(), rom=rms.tolist(),
            delta=(rms - real).tolist(), worst_abs_delta=float(np.abs(rms - real).max()),
            real_spread=float(real.max() - real.min()),
            **_corr(real, rms),
        )
    real_all = np.array([summary[b]["real"] for b in summary]).ravel()
    rom_all = np.array([summary[b]["rom"] for b in summary]).ravel()
    summary["pooled"] = dict(
        n=len(real_all), real=real_all.tolist(), rom=rom_all.tolist(),
        delta=(rom_all - real_all).tolist(),
        worst_abs_delta=float(np.abs(rom_all - real_all).max()),
        max_abs_rel=float(np.abs((rom_all - real_all) / real_all).max()),
        **_corr(real_all, rom_all),
    )
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("real", "rom", "delta")}
                      for k, v in summary.items()}, indent=2))
    if a.out:
        out = scen.resolve_out(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(rows=rows, summary=summary, m=a.m,
                                       lam=a.lam, fit=a.fit), indent=2))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
