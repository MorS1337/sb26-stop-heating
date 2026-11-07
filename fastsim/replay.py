"""Open-loop fidelity of the reduced model against a logged real rollout.

The ROM is driven only by what the real step consumed -- per-zone applied power,
air-side conductances, supply air temperatures, ambient temperature -- and its
own state. Nothing from the real temperature field is fed back after t=0, so
every error reported here is accumulated model error, not replay error.

Reported per horizon (1 h, 1 day, 10 days) and per zone:

* zone-mean temperature error, K: mean and max |error| over all zones and steps
  in the horizon, and the worst single zone;
* the same for the building's area-weighted mean, which is what the air handlers
  read as their return temperature (simulator_flexible_floor_plan.py:311-322)
  and therefore what sets supply air temperature and coil duty;
* air-side delivered heat, kWh: sum over VAVs of g (T_s - Tbar) dt, which is the
  quantity the step folds back into ``input_q`` (L470-483) and which
  ``get_zone_thermal_energy_rate`` reports.

Usage:
  starter-kit/.venv/bin/python -m fastsim.replay building_train_8/fairbanks_8 \
      --m 50,100,200,400 --kinds hybrid,krylov
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .rom import Ops, ROM

DATA = Path(__file__).resolve().parent / "data"
HORIZONS = {"1h": 12, "1d": 288, "10d": 2880}


def load_rollout(scenario: str, convection: bool = True, root: Path = DATA):
    d = root / scenario.replace("/", "__") / ("convection" if convection else "noconv")
    meta = json.loads((d / "meta.json").read_text())
    arrays = dict(np.load(d / "rollout.npz"))
    snaps = np.load(d / "snaps.npy")
    return meta, arrays, snaps


def vav_zone_weights(o: Ops) -> np.ndarray:
    """(k x n_zones) weight of each zone mean in each VAV's sensed temperature.

    The VAV reads the mean over the CVs it serves (L233-250), and every zone's
    power is spread uniformly over its CVs (building.py:1123), so with a
    zone-uniform field the sensed temperature is exactly this linear map.
    """
    return np.asarray((o.Q @ o.Zind).todense())


def run(
    scenario: str,
    m: int = 200,
    kind: str = "hybrid",
    convection: bool = True,
    n_krylov: int | None = None,
    max_steps: int | None = None,
    direct: bool = True,
    norm: str = "energy",
) -> dict:
    meta, r, snaps = load_rollout(scenario, convection)
    building = scenario.split("/")[0]
    o = Ops(building)
    t0 = time.time()
    rom = ROM(building, m=m, kind=kind, snaps=snaps, n_krylov=n_krylov, ops=o,
              direct=direct, norm=norm)
    build_s = time.time() - t0
    Vz = vav_zone_weights(o)

    q = r["q_zone"]
    G = r["G"]
    Ts = r["Ts"]
    Tamb = r["Tamb"]
    ztrue = r["ZT_out"]
    steps = min(len(Tamb), max_steps or 10**9)

    pred = np.empty((steps, o.nz))
    qa_pred = np.empty((steps, G.shape[1]))
    qa_true = np.empty((steps, G.shape[1]))
    t_step = 0.0
    rom.reset()
    rom.set_ambient(Tamb[0])
    for t in range(steps):
        t1 = time.perf_counter()
        qa = rom.step(q[t], G[t], Ts[t], Tamb[t])
        t_step += time.perf_counter() - t1
        pred[t] = rom.zone_means()
        qa_pred[t] = qa
        qa_true[t] = G[t] * (Ts[t] - Vz @ ztrue[t])

    dt = o.dt
    err = pred - ztrue[:steps]
    area = np.array([o.zcount[i] for i in range(o.nz)])
    wmean_err = (err @ area) / area.sum()
    out = {
        "scenario": scenario,
        "building": building,
        "convection": convection,
        "m": int(rom.W.shape[1]),
        "m_requested": m,
        "kind": kind,
        "n_krylov": n_krylov,
        "direct": direct,
        "norm": norm,
        "steps": steps,
        "build_seconds": build_s,
        "rho_reduced": rom.rho,
        "reduce_seconds": rom.reduce_seconds,
        "step_ms": 1000.0 * t_step / steps,
        "horizons": {},
        "zone_worst": {},
    }
    for name, n in HORIZONS.items():
        n = min(n, steps)
        e = np.abs(err[:n])
        e_w = np.abs(wmean_err[:n])
        e_zone = e.max(axis=0)
        qa_j = np.sum(qa_pred[:n] - qa_true[:n], axis=0) * dt / 3.6e6
        qa_tot = np.sum(qa_true[:n]) * dt / 3.6e6
        out["horizons"][name] = dict(
            mean_abs_K=float(e.mean()),
            max_abs_K=float(e.max()),
            rms_K=float(np.sqrt((err[:n] ** 2).mean())),
            bias_K=float(err[:n].mean()),
            worst_zone_mean_K=float(e_zone.max()),
            building_mean_abs_K=float(e_w.max()),
            building_mean_mean_K=float(e_w.mean()),
            air_heat_pred_kWh=float(np.sum(qa_pred[:n]) * dt / 3.6e6),
            air_heat_true_kWh=float(qa_tot),
            air_heat_abs_pred_kWh=float(np.abs(qa_pred[:n]).sum() * dt / 3.6e6),
            air_heat_abs_true_kWh=float(np.abs(qa_true[:n]).sum() * dt / 3.6e6),
            air_heat_abs_rel_err=float(
                (np.abs(qa_pred[:n]).sum() - np.abs(qa_true[:n]).sum())
                / max(np.abs(qa_true[:n]).sum(), 1e-9)
            ),
            air_heat_signed_rel_err=float(
                (np.sum(qa_pred[:n]) - np.sum(qa_true[:n]))
                / max(abs(np.sum(qa_true[:n])), 1e-9)
            ),
            air_heat_worst_vav_kWh=float(np.abs(qa_j).max()),
        )
    out["err"] = err
    out["t_hours"] = np.arange(steps) * dt / 3600.0
    zone_idx = np.argsort(-np.abs(err).max(axis=0))[:5]
    out["zone_worst"] = {
        o.zones[i]: dict(
            mean_abs_K=float(np.abs(err[:, i]).mean()),
            max_abs_K=float(np.abs(err[:, i]).max()),
            final_err_K=float(err[-1, i]),
        )
        for i in zone_idx
    }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scenarios", nargs="+")
    p.add_argument("--m", default="50,100,200,400")
    p.add_argument("--kinds", default="hybrid")
    p.add_argument("--krylov", type=int, default=None,
                   help="columns taken from the Krylov family in hybrid mode")
    p.add_argument("--no-convection", action="store_true")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--snap-norm", default="energy", choices=["energy", "temperature"],
                   help="metric the POD half of the basis is built in")
    p.add_argument("--no-direct", action="store_true",
                   help="drop the C^1/2 Zs / C^1/2 P state seeds to buy Krylov depth")
    p.add_argument("--out", default=None)
    p.add_argument("--err-dir", default=None,
                   help="save per-config error time series here (npz) for plotting")
    args = p.parse_args(argv)
    ms = [int(x) for x in args.m.split(",")]
    rows = []
    for scenario in args.scenarios:
        for kind in args.kinds.split(","):
            for m in ms:
                t0 = time.time()
                r = run(
                    scenario,
                    m=m,
                    kind=kind,
                    convection=not args.no_convection,
                    n_krylov=args.krylov,
                    max_steps=args.steps,
                    direct=not args.no_direct,
                    norm=args.snap_norm,
                )
                rows.append(r)
                if args.err_dir:
                    d = Path(args.err_dir)
                    d.mkdir(parents=True, exist_ok=True)
                    np.savez(
                        d / f"{r['building']}__{r['kind']}__m{r['m']}.npz",
                        err=np.stack([r["err"]]),
                        t_hours=np.arange(r["steps"]) * 300.0 / 3600.0,
                    )
                h = r["horizons"]
                print(
                    f"{scenario} {kind} m={r['m']:3d} "
                    f"1h {h['1h']['max_abs_K']:.3f} "
                    f"1d {h['1d']['max_abs_K']:.3f} "
                    f"10d {h['10d']['max_abs_K']:.3f} K max | "
                    f"mean10d {h['10d']['mean_abs_K']:.3f} "
                    f"bldg {h['10d']['building_mean_abs_K']:.3f} "
                    f"Eabs {100*h['10d']['air_heat_abs_rel_err']:+.2f}% "
                    f"| build {r["build_seconds"]+r["reduce_seconds"]:.0f}s "
                    f"step {r['step_ms']:.3f}ms ({time.time()-t0:.0f}s)"
                )
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
