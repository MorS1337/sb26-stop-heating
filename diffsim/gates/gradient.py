"""Gradient gate: the adjoint against central finite differences.

Both sides use the *smooth* twin (``quantize=False``): the objective is the
episode sum of the reward with random or unit weights, and each direction is a
random unit vector over the whole action trajectory or over one action block
(dampers, valves, AHU setpoints, recirculation, hot-water setpoint, pump head),
so a mistake in one VJP cannot hide behind the others.

    starter-kit/.venv/bin/python -m diffsim.gates.gradient --scenario \
        building_train_1/fairbanks_1 --offset-days 2 --steps 24

Exit status 0 when every direction agrees to ``--rtol`` (default 1e-6).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def blocks(nv, na):
    o = 2 * nv
    return {
        "damper": slice(0, nv),
        "valve": slice(nv, 2 * nv),
        "supply_sp": slice(o, o + na),
        "fan_sp": slice(o + na, o + 2 * na),
        "recirc": slice(o + 2 * na, o + 3 * na),
        "hws_sp": slice(o + 3 * na, o + 3 * na + 1),
        "pump_head": slice(o + 3 * na + 1, o + 3 * na + 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="building_train_1/fairbanks_1")
    ap.add_argument("--offset-days", type=int, default=0)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--start", type=int, default=288)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--directions", type=int, default=2)
    ap.add_argument("--warm-start", action="store_true", default=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from diffsim.plan import build
    from diffsim.twin import Twin
    from sbx import simload  # noqa: F401

    plan = build("corrected", a.scenario, a.start + a.steps + 2,
                 a.offset_days)
    from diffsim.replay import load_capture
    warm = load_capture("corrected", a.scenario, a.steps + a.start,
                        a.offset_days)
    twin = Twin(plan, quantize=False)
    acts = warm["acts"][a.start:a.start + a.steps].copy()
    runs = warm["runs"][a.start:a.start + a.steps].copy()
    rng = np.random.RandomState(a.seed)
    weights = np.ones(a.steps)
    nv, na = plan["nv"], plan["na"]
    bounds = dict(
        damper=(twin.vav_min_damper, np.ones(nv)),
        valve=(np.zeros(nv), np.ones(nv)),
        supply_sp=(np.full(na, 285.0), np.full(na, 305.0)),
        fan_sp=(np.full(na, 200.0), np.full(na, 2000.0)),
        recirc=(np.zeros(na), plan["ahu_max_recirc"]),
        hws_sp=(np.array([310.0]), np.array([350.0])),
        pump_head=(np.array([0.0]), np.array([20.0 / 0.0980665])),
    )

    def objective(acts_):
        rew, pen, viol, _, _ = twin.forward(acts_, runs, keep_tape=False,
                                           i0=a.start)
        return float(np.dot(weights, rew))

    rew, pen, viol, tape, _ = twin.forward(acts, runs, keep_tape=True, i0=a.start)
    J = float(np.dot(weights, rew))
    d_acts, d_state = twin.backward(tape, d_reward=weights)

    out = {"scenario": a.scenario, "steps": a.steps, "start": a.start,
           "J": J, "eps": a.eps, "dirs": []}
    worst = 0.0
    blk = blocks(nv, na)
    for name, sl in [("all", slice(0, twin.n_act))] + list(blk.items()):
        for d in range(a.directions):
            v = np.zeros_like(acts)
            v[:, sl] = rng.randn(a.steps, len(range(*sl.indices(twin.n_act))))
            nrm = np.sqrt(np.sum(v * v))
            if nrm == 0.0:
                continue
            v /= nrm
            # the step is scaled to the block's own magnitude: a damper moves
            # in units of 1, a temperature in K (a fixed 1e-4 on the hot-water
            # setpoint would be a 1e-6 relative step and the FD would drown in
            # round-off, while the same absolute step on the damper would be a
            # 0.01 % perturbation of a 1.0 command)
            eps = a.eps * max(0.1, float(np.max(np.abs(acts[:, sl]))))
            Jp = objective(acts + eps * v)
            Jm = objective(acts - eps * v)
            fd = (Jp - Jm) / (2.0 * eps)
            an = float(np.sum(d_acts * v))
            rel = abs(fd - an) / max(abs(fd), abs(an), 1e-12)
            worst = max(worst, rel)
            row = {"block": name, "dir": d, "fd": fd, "adjoint": an,
                   "rel": rel, "eps": eps}
            out["dirs"].append(row)
            print(f"{name:10s} dir {d}: central {fd:+.10e} adjoint {an:+.10e} "
                  f"rel {rel:.3e}")
    out["worst"] = worst
    ok = worst <= a.rtol
    print(f"worst relative error {worst:.3e} (rtol {a.rtol:g})")
    print("GATE PASS" if ok else "GATE FAIL")
    if a.out:
        p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(out, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    import os as _os
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # the C kernel's teardown corrupts the glibc heap on exit (a segfault
    # after the results are written, WSL2/glibc only); skip it
    _os._exit(int(_rc))
