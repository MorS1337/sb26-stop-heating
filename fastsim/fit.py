"""Exact replay traces and the static bias correction fitted to them.

Two things live here.

``exact_replay`` runs the real linear algebra (the extracted factorization) with
the logged inputs, keeping the full field, and records the reduced coordinates
of each true state. It is the reference the reduction is measured against, and
it validates the extraction: replaying the logged inputs through the exact
operators reproduces the logged zone means to ~1e-3 K over 10 days, so every
error in ``fastsim.replay`` is reduction error and nothing else.

``fit_bias`` measures the one-step residual of the reduced map along that true
trajectory and fits one correction vector to its mean.

Why a bias and not a better basis: the residual is not truncation. Fidelity is
unchanged from m = 100 to m = 400 (1.23 -> 1.17 K over 10 days) while the
steady-state response to a local heat input improves 31 % -> 4.6 %, so the
surviving error is a small persistent imbalance, not missing detail. A constant
imbalance integrated over 2880 steps is exactly a ramp, which is the shape the
error has (0.002 K at 1 h, 0.45 at 1 day, 1.17 at 10 days). One vector removes
it. It is fitted on the first half of the episode and scored on the whole
episode, so the second half is a holdout.

Usage:
  starter-kit/.venv/bin/python -m fastsim.fit building_train_8/fairbanks_8 --m 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .replay import HORIZONS, load_rollout, vav_zone_weights
from .rom import Ops, ROM


def exact_replay(o: Ops, r, steps: int | None = None, rom: ROM | None = None):
    """Runs the exact step from the logged inputs; returns reduced coordinates.

    Args:
      o: extracted operators.
      r: logged rollout arrays.
      steps: number of steps to replay.
      rom: if given, also records its one-step prediction error along the way.

    Returns:
      dict with ``s`` (steps, m) and ``u_m`` (steps, ng) of the true states, the
      zone means produced, and (optionally) the ROM's one-step residuals.
    """
    n = min(steps or 10**9, len(r["Tamb"]))
    k = o.k
    AinvP = o.solve(o.P.toarray())
    Sa = np.asarray(o.Q @ AinvP)
    Tm = o.T0_m.copy()
    T = o.T0[o.solved].copy()
    W = None if rom is None else rom.W
    m = 0 if W is None else W.shape[1]
    out = {
        "s": np.empty((n, m)),
        "u_m": np.empty((n, o.ng)),
        "zone": np.empty((n, o.nz)),
        "zone_next": np.empty((n, o.nz)),
        "rom_step_err": np.empty((n, o.nz)) if rom is not None else None,
        "rom_b": np.empty((n, m)) if rom is not None else None,
    }
    for t in range(n):
        G, Ts = r["G"][t], r["Ts"][t]
        b = o.c * T + o.Zs @ r["q_zone"][t] + (o.d + o.g_amb) * r["Tamb"][t] + o.mc * Tm
        y = o.solve(b)
        qa = np.linalg.solve(np.eye(k) + G[:, None] * Sa, G * (Ts - o.Q @ y))
        T_next = y + AinvP @ qa
        Tm = o.retain * Tm + o.gain * T_next
        if rom is not None:
            s_true = W.T @ (o.sqrtc * T)
            u_true = o.Zm @ Tm
            b_red = (
                rom.Ac @ s_true
                + rom.Aq @ r["q_zone"][t]
                + rom.dv * r["Tamb"][t]
                + rom.Mr @ u_true
            )
            rhs = G * (Ts - rom.Qq @ b_red)
            qa_red = np.linalg.solve(np.eye(k) + G[:, None] * rom.Sa, rhs)
            s_red = b_red + rom.Pr @ qa_red
            out["rom_step_err"][t] = rom.Zo @ s_red - o.zone_means(T_next)
            out["rom_b"][t] = s_red - (W.T @ (o.sqrtc * T_next))
        out["s"][t] = o.sqrtc @ T if W is None else W.T @ (o.sqrtc * T)
        out["u_m"][t] = o.Zm @ Tm
        out["zone"][t] = o.zone_means(T)
        out["zone_next"][t] = o.zone_means(T_next)
        T = T_next
    return out


def residuals(scenario: str, m: int = 200, kind: str = "hybrid",
              convection: bool = True) -> dict:
    """One-step residuals of the reduced map along the true trajectory."""
    building = scenario.split("/")[0]
    meta, r, snaps = load_rollout(scenario, convection)
    o = Ops(building)
    rom = ROM(building, m=m, kind=kind, snaps=snaps, ops=o)
    ex = exact_replay(o, r, rom=rom)
    return dict(rom=rom, o=o, r=r, ex=ex, meta=meta, snaps=snaps)


def fit_zone_heat(scenario: str, m: int = 200, kind: str = "hybrid",
                  fit_frac: float = 0.5, convection: bool = True) -> dict:
    """Fits the residual as an extra per-zone heat input (n_zones parameters).

    The one-step residual of the reduced map is a vector in R^m. Parameterising
    it as ``Aq @ alpha`` is the physically meaningful restriction: it says the
    unresolved part of the step behaves like a small unmodelled heat gain or
    loss in each zone, which is what a truncated representation of the room's
    near-diffuser response looks like. It is 30 numbers instead of 200, and it
    cannot inject a component the heat-input dynamics could not produce, which
    matters because a raw R^m mean residual is dominated by directions the
    slow loop amplifies without bound (measured: 943 K).
    """
    d = residuals(scenario, m, kind, convection)
    rom, o, r, ex = d["rom"], d["o"], d["r"], d["ex"]
    n = len(r["Tamb"])
    fit_n = int(fit_frac * n)
    alpha, *_ = np.linalg.lstsq(rom.Aq, ex["rom_b"][:fit_n].mean(axis=0), rcond=None)
    hold, *_ = np.linalg.lstsq(rom.Aq, ex["rom_b"][fit_n:].mean(axis=0), rcond=None)
    cos = float(
        hold @ alpha / (np.linalg.norm(hold) * np.linalg.norm(alpha) + 1e-30)
    )
    return dict(d, alpha=alpha, bias=rom.Aq @ alpha, fit_n=fit_n, n=n,
                holdout_cosine=cos)


def score(rom: ROM, o: Ops, r, bias: np.ndarray | None, steps: int | None = None,
          T_init: float | None = None) -> dict:
    """Open-loop replay of the reduced model, optionally with a bias term.

    ``T_init`` starts from a uniform field of that temperature, which is what the
    simulator's ``reset`` leaves in the building; the recorded initial zone means
    are used for it so a holdout strip with a different ``initial_temp`` starts
    where its own rollout started.
    """
    n = min(steps or 10**9, len(r["Tamb"]))
    pred = np.empty((n, o.nz))
    qa_pred = np.empty((n, o.k))
    qa_true = np.empty((n, o.k))
    vz = vav_zone_weights(o)
    if T_init is None:
        rom.reset()
    else:
        rom.reset(np.full(o.nfull, float(T_init)), np.full(o.n, float(T_init)))
    for t in range(n):
        b = (
            rom.Ac @ rom.s
            + rom.Aq @ r["q_zone"][t]
            + rom.dv * r["Tamb"][t]
            + rom.Mr @ rom.u_m
        )
        if bias is not None:
            b = b + bias
        G, Ts = r["G"][t], r["Ts"][t]
        rhs = G * (Ts - rom.Qq @ b)
        qa = np.linalg.solve(np.eye(G.size) + G[:, None] * rom.Sa, rhs)
        rom.s = b + rom.Pr @ qa
        rom.u_m = rom.r * rom.u_m + rom.g * (rom.Zm_u @ rom.s)
        pred[t] = rom.Zo @ rom.s
        qa_pred[t] = qa
        qa_true[t] = G * (Ts - vz @ r["ZT_out"][t])
    err = pred - r["ZT_out"][:n]
    area = o.zcount
    wmean = (err @ area) / area.sum()
    out = {}
    for name, h in HORIZONS.items():
        h = min(h, n)
        e = np.abs(err[:h])
        out[name] = dict(
            max_abs_K=float(e.max()),
            mean_abs_K=float(e.mean()),
            bias_K=float(err[:h].mean()),
            building_mean_abs_K=float(np.abs(wmean[:h]).max()),
        )
    dt = o.dt
    out["energy"] = dict(
        air_heat_pred_kWh=float(qa_pred.sum() * dt / 3.6e6),
        air_heat_true_kWh=float(qa_true.sum() * dt / 3.6e6),
        air_heat_abs_pred_kWh=float(np.abs(qa_pred).sum() * dt / 3.6e6),
        air_heat_abs_true_kWh=float(np.abs(qa_true).sum() * dt / 3.6e6),
        air_heat_abs_rel_err=float(
            (np.abs(qa_pred).sum() - np.abs(qa_true).sum())
            / max(np.abs(qa_true).sum(), 1e-9)
        ),
        air_heat_signed_rel_err=float(
            (qa_pred.sum() - qa_true.sum()) / max(abs(qa_true.sum()), 1e-9)
        ),
    )
    out["err"] = err
    return out


def _vav_zone(o: Ops) -> np.ndarray:
    return np.asarray((o.Q @ o.Zind).todense())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scenarios", nargs="+")
    p.add_argument("--m", default="100,200,400")
    p.add_argument("--kinds", default="hybrid")
    p.add_argument("--fit-frac", type=float, default=0.5)
    p.add_argument("--no-convection", action="store_true")
    p.add_argument("--scales", default="0,0.25,0.33,0.5,0.75,1.0",
                   help="multipliers on the fitted zone-heat correction to test")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    rows = []
    for scenario in a.scenarios:
        for kind in a.kinds.split(","):
            for m in [int(x) for x in a.m.split(",")]:
                r = fit_zone_heat(scenario, m, kind, a.fit_frac,
                                  convection=not a.no_convection)
                rom, o, rl, ex = r["rom"], r["o"], r["r"], r["ex"]
                one = np.abs(ex["rom_step_err"]).max()
                base = score(rom, o, rl, None)
                corr = score(rom, o, rl, r["bias"])
                scaled = {}
                for lam in [float(x) for x in a.scales.split(",")]:
                    s = score(rom, o, rl, lam * r["bias"])
                    half = r["fit_n"] // 2
                    scaled[lam] = dict(
                        d5_max=float(np.abs(s["err"][: r["fit_n"]]).max()),
                        d5_mean=float(np.abs(s["err"][: r["fit_n"]]).mean()),
                        hold_max=float(np.abs(s["err"][r["fit_n"] :]).max()),
                        hold_mean=float(np.abs(s["err"][r["fit_n"] :]).mean()),
                        d10_max=s["10d"]["max_abs_K"],
                        d10_mean=s["10d"]["mean_abs_K"],
                        d10_bias=s["10d"]["bias_K"],
                    )
                base_half = dict(
                    d5_max=float(np.abs(base["err"][: r["fit_n"]]).max()),
                    d5_mean=float(np.abs(base["err"][: r["fit_n"]]).mean()),
                    hold_max=float(np.abs(base["err"][r["fit_n"] :]).max()),
                    hold_mean=float(np.abs(base["err"][r["fit_n"] :]).mean()),
                )
                rows.append(dict(
                    scenario=scenario, kind=kind, m=int(rom.W.shape[1]),
                    one_step_max_K=float(one),
                    one_step_mean_K=float(np.abs(ex["rom_step_err"]).mean()),
                    alpha=dict(zip(o.zones, np.round(r["alpha"], 2).tolist())),
                    bias_norm=float(np.linalg.norm(r["bias"])),
                    bias_holdout_cosine=r["holdout_cosine"],
                    unbiased={k: v for k, v in base.items() if k not in ("err", "final_hours")},
                    unbiased_halves=base_half,
                    corrected={k: v for k, v in corr.items() if k not in ("err", "final_hours")},
                    scaled=scaled,
                ))
                print("  scale  d0-5 max/mean   d5-10 max/mean   10d max/mean  bias")
                for lam, v in scaled.items():
                    print(f"  {lam:5.2f}  {v['d5_max']:.3f}/{v['d5_mean']:.3f}     "
                          f"{v['hold_max']:.3f}/{v['hold_mean']:.3f}      "
                          f"{v['d10_max']:.3f}/{v['d10_mean']:.3f}  {v['d10_bias']:+.3f}")
                print(
                    f"{scenario} {kind} m={rom.W.shape[1]:3d} one-step max {one:.2e} K "
                    f"mean {np.abs(ex['rom_step_err']).mean():.2e} | "
                    f"unbiased 1d {base['1d']['max_abs_K']:.3f} 10d {base['10d']['max_abs_K']:.3f} | "
                    f"zone-heat 1d {corr['1d']['max_abs_K']:.3f} "
                    f"10d {corr['10d']['max_abs_K']:.3f} mean {corr['10d']['mean_abs_K']:.4f} K "
                    f"| alpha |.|={np.linalg.norm(r['alpha']):.3g} W holdout cos "
                    f"{r['holdout_cosine']:.3f}"
                )
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
