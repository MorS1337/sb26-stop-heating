"""Extracts the exact per-timestep thermal operators from the real solver.

Nothing is re-derived by hand: every array here is read off the live
``FiniteDifferenceSolver`` instance that the real environment built, so the ROM
reproduces the shipped linear algebra up to the reduced basis.

The step the solver implements (finite_difference_solver.py:968-1025) is

    A(h) = D(total_conductance + C/dt + M_c) - K + D(g_amb(h))        [n x n]
    b    = (C/dt) T + q + T_amb (d + g_amb(h)) + M_c T_m
    y    = A^-1 b
    q_a  = (I + G S)^-1 G (T_s - Q^T y),   S = Q^T A^-1 P
    T'   = y + A^-1 P q_a
    T_m' = r . T_m + g . T'

with ``C/dt`` = _capacitance_rate, ``d`` = _dirichlet_conductance (coupling to
the pinned exterior CVs), ``g_amb`` = ambient_conductance(h), ``M_c`` =
_mass_coupling, (r, g) = _mass_retained/_mass_gained, P = _applied,
Q^T = _sensed, G = diag(air_side_conductance).

Outputs, per building (``fastsim/ops/<building>/``):

  meta.json          grid shape, zone names, timestep, convection coefficient
  A.npz              the assembled timestep matrix at that h (csc)
  vectors.npz        solved indices, dt, C/dt, d, g_amb, M_c, r, g, T0, T0_m
  applied.npz        P, (n_solved x k) csc, signed so it sums to +1 per column
  sensed.npz         Q^T, (k x n_solved) csr
  zones.npz          CSR of zone CV positions within the solved vector

Usage:
  starter-kit/.venv/bin/python -m fastsim.extract building_train_8/fairbanks_8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import scen

OPS = Path(__file__).resolve().parent / "ops"


def physical_zones(building) -> list[str]:
    from smart_buildings.smart_control.simulator import constants  # noqa: PLC0415

    return [z for z in building.room_dict if not constants.is_non_physical_space(z)]


def zone_csr(building, zones: list[str], pos_of_full: np.ndarray, n_solved: int):
    """CSR (n_zones x n_solved) averaging operator over each zone's CVs."""
    positions, ptr = [], [0]
    for z in zones:
        coords = np.asarray(building.room_dict[z], dtype=np.intp).reshape(-1, 2)
        flat = coords[:, 0] * building.temp.shape[1] + coords[:, 1]
        p = pos_of_full[flat]
        if (p < 0).any():
            raise ValueError(f"zone {z} contains a CV that is not solved for")
        positions.append(p)
        ptr.append(ptr[-1] + p.size)
    ind = np.concatenate(positions)
    data = np.concatenate([np.full(n, 1.0 / n) for n in np.diff(ptr)])
    return sp.csr_matrix((data, ind, np.asarray(ptr)), shape=(len(zones), n_solved))


def extract(scenario: str, sim: str = "shipped", out_root: Path | None = None) -> Path:
    env, _ = scen.build_env(scenario, sim)
    simulator = env.building.simulator
    building = simulator.building
    solver = simulator._solver
    ts = simulator.current_timestamp
    h = float(simulator._weather_controller.get_air_convection_coefficient(ts))

    n_full = building.temp.size
    ncols = building.temp.shape[1]
    solved = np.asarray(solver._solved)
    pos_of_full = np.full(n_full, -1, dtype=np.int64)
    pos_of_full[solved] = np.arange(solved.size)

    zones = physical_zones(building)
    zmat = zone_csr(building, zones, pos_of_full, solved.size)

    A = solver.matrix(h)
    if not sp.issparse(A):
        raise TypeError("solver.matrix did not return a sparse matrix")
    A = sp.csc_matrix(A)

    applied = sp.csc_matrix(solver._applied)
    sensed = sp.csr_matrix(solver._sensed)  # (k x n_solved)

    dt = float(solver._capacitance_rate.size and simulator.time_step_sec)
    vectors = dict(
        solved=solved,
        dt=dt,
        capacitance_rate=np.asarray(solver._capacitance_rate),
        dirichlet_conductance=np.asarray(solver._dirichlet_conductance),
        ambient_conductance=np.asarray(solver.ambient_conductance(h)),
        mass_coupling=np.asarray(solver._mass_coupling),
        mass_retained=np.asarray(solver._mass_retained),
        mass_gained=np.asarray(solver._mass_gained),
        T0=np.asarray(building.temp, dtype=np.float64).reshape(-1),
        T0_m=np.asarray(solver._mass_temp, dtype=np.float64),
    )

    out = (out_root or OPS) / scenario.split("/")[0]
    out.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out / "A.npz", A)
    sp.save_npz(out / "applied.npz", applied)
    sp.save_npz(out / "sensed.npz", sensed)
    sp.save_npz(out / "zones.npz", zmat)
    np.savez_compressed(out / "vectors.npz", **vectors)
    (out / "meta.json").write_text(
        json.dumps(
            dict(
                scenario=scenario,
                sim=sim,
                grid=list(building.temp.shape),
                n_full=n_full,
                n_solved=int(solved.size),
                n_zones=len(zones),
                zones=zones,
                n_air_terms=int(applied.shape[1]),
                convection_coefficient=h,
                timestep_sec=dt,
                start_timestamp=str(ts),
                cv_size_m=float(building.cv_size_cm) / 100.0,
                floor_height_m=float(building.floor_height_cm) / 100.0,
                A_nnz=int(A.nnz),
            ),
            indent=2,
        )
    )
    print(
        f"{scenario}: grid {tuple(building.temp.shape)} n_solved {solved.size} "
        f"zones {len(zones)} air terms {applied.shape[1]} A nnz {A.nnz} -> {out}"
    )
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scenarios", nargs="+")
    p.add_argument("--sim", default="shipped")
    args = p.parse_args(argv)
    scen.pin_blas()
    scen.boot(args.sim)
    for s in args.scenarios:
        extract(s, args.sim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
