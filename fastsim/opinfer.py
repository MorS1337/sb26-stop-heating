"""Operator inference: ridge refit of the four reduced maps on exact transitions.

The projected model's remaining error is not truncation of the *fields* (a
projected state still reproduces zone means to 2e-3 K over 10 days) but the small
coherent mismatch between the projected forcing map and the real one:
``W^T C^1/2 A^-1 (C/dt . T + ...)``. Each of those maps is linear, so the
mismatch can be fitted directly instead of being carried as modelling error.

Right-hand side and regressors are exactly the projected model's:

    b_k = W^T C^1/2 A^-1 ( C/dt T_k + q_k + T_amb (d+g) + M_c T_m,k )
        ~= Ac s_k + Aq q_k + dv T_amb,k + Mr u_m,k                     (11)

``b_k`` is read off the *exact* solve along the real trajectory (no air side:
the air-side contribution is subtracted by construction, because ``b`` is the
pre-Woodbury solution), the regressors are the exact reduced state and the
logged inputs, and the four maps are fitted by ridge least squares. Fitting the
residual relative to the projected maps (``b - b_proj``) rather than the maps
themselves keeps the fit a correction: with the penalty on the correction it
cannot wander into a different, unstable operator, and the target below is what
an honest validation measures.

Validation is a **different weather strip of the same building**, with a basis
built only from the fitting strip's fields, so nothing from the validation strip
enters the model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .replay import load_rollout
from .rom import Ops, ROM


def exact_transitions(o: Ops, rom: ROM, r, steps: int | None = None,
                      T_init: float | None = None) -> dict:
    """Runs the exact step from the logged inputs and records (11)'s two sides.

    ``T_init`` starts the field uniform at that temperature (what the simulator's
    reset leaves); it defaults to the field the operators were extracted from, so
    a strip with a different ``initial_temp`` still traces a trajectory in the
    same regime as its own rollout.
    """
    n = min(steps or 10**9, len(r["Tamb"]))
    W, sc, k = rom.W, o.sqrtc, o.k
    AinvP = o.solve(o.P.toarray())
    Sa = np.asarray(o.Q @ AinvP)
    if T_init is None:
        T = o.T0[o.solved].copy()
        Tm = o.T0_m.copy()
    else:
        T = np.full(o.n, float(T_init))
        Tm = np.where(o.mc > 0.0, float(T_init), 0.0)
    m, ng = W.shape[1], o.ng
    S = np.empty((n, m))
    U = np.empty((n, ng))
    Q = np.asarray(r["q_zone"][:n], dtype=np.float64)
    Tamb = np.asarray(r["Tamb"][:n], dtype=np.float64)
    B = np.empty((n, m))
    QA = np.empty((n, k))
    for t in range(n):
        S[t] = W.T @ (sc * T)  # s_k: the state entering the step
        U[t] = o.Zm @ Tm  # u_m,k: the mass node entering the step
        b_full = o.c * T + o.Zs @ Q[t] + (o.d + o.g_amb) * Tamb[t] + o.mc * Tm
        y = o.solve(b_full)
        B[t] = W.T @ (sc * y)  # b_k: the exact pre-air-side solution
        G, Ts = r["G"][t], r["Ts"][t]
        qa = np.linalg.solve(np.eye(k) + G[:, None] * Sa, G * (Ts - o.Q @ y))
        T = y + AinvP @ qa
        Tm = o.retain * Tm + o.gain * T
        QA[t] = qa
    return dict(S=S, U=U, Q=Q, Tamb=Tamb, B=B, QA=QA)


def stack_maps(Ac, Aq, dv, Mr) -> np.ndarray:
    """(features x m) form of [Ac | Aq | dv | Mr], features in ``design`` order."""
    return np.vstack([Ac, Aq.T, np.asarray(dv).reshape(1, -1), Mr.T])


def design(S, U, Q, Tamb):
    """Regressor matrix [s | q | T_amb | u_m] for (11)."""
    return np.hstack([S, Q, Tamb.reshape(-1, 1), U])


def fit_maps(o: Ops, rom: ROM, r, lam: float = 1e-6, steps: int | None = None,
             verbose: bool = False) -> dict:
    """Ridge fit of Ac, Aq, dv, Mr on a rollout; returns the fitted matrices."""
    t0v = np.asarray(r["ZT_in"][0], dtype=np.float64)
    T_init = float(t0v.mean()) if np.ptp(t0v) < 1e-9 else None
    tr = exact_transitions(o, rom, r, steps, T_init=T_init)
    return fit_from_transitions(o, rom, tr, lam=lam, verbose=verbose)


def fit_from_transitions(o: Ops, rom: ROM, tr: dict, lam: float = 1e-6,
                         verbose: bool = False) -> dict:
    """Ridge fit of Ac, Aq, dv, Mr from precomputed transitions."""
    t0 = time.time()
    X = design(tr["S"], tr["U"], tr["Q"], tr["Tamb"])
    # Target of the projection is b_proj = Ac s + Aq q + dv T + Mr u, so the
    # correction is fitted rather than the operator itself (see the docstring).
    proj = X @ stack_maps(rom.Ac, rom.Aq, rom.dv, rom.Mr)
    Y = tr["B"] - proj
    sc = X.std(axis=0)
    sc[sc == 0] = 1.0
    Xs = X / sc
    gram = Xs.T @ Xs
    ridge = lam * np.trace(gram) / gram.shape[0]
    coef = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), Xs.T @ Y) / sc[:, None]
    d = coef
    m = rom.W.shape[1]
    m_new = dict(
        Ac=rom.Ac + d[:m],
        Aq=rom.Aq + d[m : m + o.nz].T,
        dv=rom.dv + d[m + o.nz],
        Mr=rom.Mr + d[m + o.nz + 1 :].T,
    )
    m_new["rho"] = float(np.abs(np.linalg.eigvals(m_new["Ac"])).max())
    m_new["seconds"] = time.time() - t0
    m_new["rel_correction"] = {
        "Ac": float(np.linalg.norm(d[:m]) / np.linalg.norm(rom.Ac)),
        "Aq": float(np.linalg.norm(d[m : m + o.nz]) / max(np.linalg.norm(rom.Aq), 1e-30)),
        "dv": float(np.linalg.norm(d[m + o.nz]) / max(np.linalg.norm(rom.dv), 1e-30)),
        "Mr": float(np.linalg.norm(d[m + o.nz + 1 :]) / max(np.linalg.norm(rom.Mr), 1e-30)),
    }
    if verbose:
        print(f"  fit m={m}: {m_new['seconds']:.1f}s rho {m_new['rho']:.6f} "
              f"corr {m_new['rel_correction']}")
    return m_new


def apply_maps(rom: ROM, maps: dict) -> None:
    rom.Ac, rom.Aq, rom.dv, rom.Mr = maps["Ac"], maps["Aq"], maps["dv"], maps["Mr"]


def build(
    building: str,
    m: int = 200,
    basis_scenario: str | None = None,
    fit: bool = True,
    fit_scenario: str | None = None,
    lam: float = 1e-6,
    kind: str = "hybrid",
    steps: int | None = None,
    verbose: bool = True,
) -> tuple[ROM, Ops, dict | None]:
    """Builds a ROM whose basis comes from ``basis_scenario`` and whose maps are
    optionally refitted on it."""
    basis_scenario = basis_scenario or f"{building}/fairbanks_8"
    fit_scenario = fit_scenario or basis_scenario
    _, _, snaps = load_rollout(basis_scenario)
    o = Ops(building)
    rom = ROM(building, m=m, kind=kind, snaps=snaps, ops=o)
    maps = None
    if fit:
        _, r, _ = load_rollout(fit_scenario)
        maps = fit_maps(o, rom, r, lam=lam, steps=steps, verbose=verbose)
        apply_maps(rom, maps)
        rom.reset()
    return rom, o, maps


def main(argv=None) -> int:
    import argparse  # noqa: PLC0415

    from .fit import score
    from .replay import HORIZONS

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", nargs="+", required=True,
                   help="fit_scenario:holdout_scenario, e.g. "
                        "building_train_8/fairbanks_8:building_train_8/denver_6")
    p.add_argument("--m", default="200")
    p.add_argument("--lam", default="1e-6,1e-4,1e-2")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    rows = []
    for pair in a.pairs:
        fit_sc, hold = pair.split(":")
        building = fit_sc.split("/")[0]
        for m in [int(x) for x in a.m.split(",")]:
            _, _, snaps = load_rollout(fit_sc)
            o = Ops(building)
            rom = ROM(building, m=m, kind="hybrid", snaps=snaps, ops=o)
            _, r_fit, _ = load_rollout(fit_sc)
            _, r_hold, _ = load_rollout(hold)
            t0 = r_hold["ZT_in"][0]
            uniform = float(np.ptp(t0)) < 1e-9
            T_init = float(t0.mean())
            base = score(rom, o, r_hold, None, steps=a.steps,
                         T_init=T_init if uniform else None)
            pristine = dict(Ac=rom.Ac.copy(), Aq=rom.Aq.copy(),
                            dv=rom.dv.copy(), Mr=rom.Mr.copy())
            t0v = np.asarray(r_fit["ZT_in"][0], dtype=np.float64)
            tr = exact_transitions(
                o, rom, r_fit, a.steps,
                T_init=float(t0v.mean()) if np.ptp(t0v) < 1e-9 else None,
            )
            for lam in [float(x) for x in a.lam.split(",")]:
                apply_maps(rom, pristine)  # fit the correction from the projection
                maps = fit_from_transitions(o, rom, tr, lam=lam, verbose=True)
                apply_maps(rom, maps)
                s = score(rom, o, r_hold, None, steps=a.steps,
                          T_init=T_init if uniform else None)
                rows.append(dict(
                    building=building, m=int(rom.W.shape[1]), lam=lam,
                    fit=fit_sc, holdout=hold, uniform_init=uniform,
                    rho=maps["rho"], rel=maps["rel_correction"],
                    projected={h: base[h] for h in HORIZONS},
                    inferred={h: s[h] for h in HORIZONS},
                    energy=s["energy"],
                ))
                print(
                    f"{building} m={rom.W.shape[1]} lam={lam:g} | projected "
                    f"1h {base['1h']['max_abs_K']:.4f} 1d {base['1d']['max_abs_K']:.3f} "
                    f"10d {base['10d']['max_abs_K']:.3f} (mean {base['10d']['mean_abs_K']:.3f}) "
                    f"| inferred 1h {s['1h']['max_abs_K']:.4f} "
                    f"1d {s['1d']['max_abs_K']:.3f} 10d {s['10d']['max_abs_K']:.3f} "
                    f"(mean {s['10d']['mean_abs_K']:.3f}) rho {maps['rho']:.4f} "
                    f"[holdout {hold}]"
                )
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
