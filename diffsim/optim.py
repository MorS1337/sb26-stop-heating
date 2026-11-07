"""Trajectory optimisation through the differentiable twin.

    starter-kit/.venv/bin/python -m diffsim.optim anchor --iters 40

The decision variable is the whole action trajectory (the 92 continuous native
commands per step), optimised with projected Adam inside the action bounds the
environment's normalisers enforce.  Two information regimes:

``full``  the objective is the true episode reward (comfort, energy, carbon)
          against the realised occupancy and weather: everything the simulator
          knows is available to the planner;
``band``  comfort is a hard constraint -- every zone inside its schedule band at
          every comfort-mode step -- so the plan needs no per-zone occupancy
          information at all; it is enforced as a quadratic penalty and the
          trajectory is scored with the true reward afterwards.

Everything the optimiser sees is the smooth twin (no float32 reward
quantisation, see ``Twin``); every number *reported* comes from
:mod:`diffsim.replay`, i.e. the fused kernel and the simulator's own reward.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results")
SCRATCH = os.path.join(ROOT, "scratch", "ExactOpt")


# --------------------------------------------------------------------------- #
def bounds_for(plan, twin):
    nv, na = plan["nv"], plan["na"]
    lo = np.empty(twin.n_act)
    hi = np.empty(twin.n_act)
    o = 2 * nv
    lo[:nv] = twin.vav_min_damper
    hi[:nv] = 1.0
    lo[nv:2 * nv] = 0.0
    hi[nv:2 * nv] = 1.0
    lo[o:o + na] = 285.0
    hi[o:o + na] = 305.0
    lo[o + na:o + 2 * na] = plan["ahu_min_sp"]
    hi[o + na:o + 2 * na] = 2000.0
    lo[o + 2 * na:o + 3 * na] = 0.0
    hi[o + 2 * na:o + 3 * na] = plan["ahu_max_recirc"]
    lo[o + 3 * na] = 310.0
    hi[o + 3 * na] = 350.0
    lo[o + 3 * na + 1] = 0.0
    hi[o + 3 * na + 1] = 20.0 / (9.8 * 1000.0 / 1e5)  # 20 bar of head, in m
    return lo, hi


def project(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def run_regime(plan, warm, regime, iters, lr, mu=200.0, log=None,
               init=None, seed=0):
    """Projected Adam on the whole trajectory. Returns (theta, history)."""
    from diffsim.twin import Twin

    twin = Twin(plan, quantize=False)
    lo, hi = bounds_for(plan, twin)
    theta = (warm["acts"].copy() if init is None else init.copy())
    theta = project(theta, lo, hi)
    runs = warm["runs"]
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-12
    hist = []
    best = None
    for it in range(iters):
        t0 = time.process_time()
        rew, pen, viol, tape, _ = twin.forward(theta, runs)
        J = -float(rew.sum()) + (mu * float(pen.sum()) if regime == "band" else 0.0)
        d_acts, _ = twin.backward(
            tape, d_reward=-np.ones(len(rew)),
            penalty_mu=(mu if regime == "band" else 0.0))
        # gradient clipping keeps the first steps from overshooting
        g = d_acts
        gn = float(np.sqrt(np.sum(g * g)))
        scale = min(1.0, 30.0 / max(gn, 1e-30))
        g = g * scale
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g * g
        mh = m / (1.0 - b1 ** (it + 1))
        vh = v / (1.0 - b2 ** (it + 1))
        step = lr * mh / (np.sqrt(vh) + eps)
        theta = project(theta - step, lo, hi)
        row = dict(iter=it, J=J, reward=float(rew.sum()),
                   penalty=float(pen.sum()), viol=float(viol.max()),
                   gnorm=gn, lr=lr, cpu=time.process_time() - t0)
        if regime == "band":
            # the constraint is what matters: keep the plan with the fewest
            # violations, then the lowest cost
            key = (round(row["penalty"], 9), row["J"])
        else:
            key = (row["J"],)
        if best is None or key < best[0]:
            best = (key, theta.copy(), row)
        hist.append(row)
        if log:
            log(row)
    return best[1], hist, best[2]


# --------------------------------------------------------------------------- #
SCENARIOS = [
    # (name, sim, scenario, offset_days)
    ("anchor_corrected", "corrected", "building_train_1/fairbanks_1", 0),
    ("anchor_shipped", "shipped", "building_train_1/fairbanks_1", 0),
    ("phoenix5_corrected", "corrected", "building_train_1/phoenix_5", 0),
    ("singapore8_corrected", "corrected", "building_train_1/singapore_8", 0),
    ("atlanta3_corrected", "corrected", "building_train_2/atlanta_3", 0),
    ("denver5_corrected", "corrected", "building_train_7/denver_5", 0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["anchor", "scenario", "compare"])
    ap.add_argument("--name", default="anchor_corrected")
    ap.add_argument("--sim", default=None)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--offset-days", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2016)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--mu", type=float, default=200.0)
    ap.add_argument("--regimes", default="full,band")
    ap.add_argument("--restart", action="store_true",
                    help="run from a different initialisation (lower fan "
                         "static pressure and hot-water setpoint for the first "
                         "half of the episode)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    from diffsim import optim as me

    if a.cmd in ("anchor", "scenario"):
        name = a.name
        sim, scenario, off = a.sim, a.scenario, a.offset_days
        if sim is None or scenario is None:
            for n, s, sc, o in SCENARIOS:
                if n == name:
                    sim, scenario, off = s, sc, o
        from diffsim.plan import build
        from diffsim.replay import load_capture, replay

        warm = load_capture(sim, scenario, a.steps, off)
        plan = build(sim, scenario, a.steps + 2, off)
        print(f"== {name}: {sim} {scenario} off={off} steps={a.steps} "
              f"warm total {warm['rewards'].sum():.6f}", flush=True)
        os.makedirs(os.path.join(SCRATCH, "optim"), exist_ok=True)
        tag = a.tag or f"{name}_{a.steps}"
        report = {"name": name, "sim": sim, "scenario": scenario,
                  "steps": a.steps, "offset_days": off,
                  "warm_total": float(warm["rewards"].sum()), "runs": {}}
        for regime in a.regimes.split(","):
            t0 = time.time()
            init = None
            if a.restart and regime == "full":
                # a different initialisation: the warm start with the AHU
                # static pressure and the hot-water setpoint pushed to their
                # lower bounds over the first half of the episode
                init = warm["acts"].copy()
                nv, na = plan["nv"], plan["na"]
                o = 2 * nv
                half = a.steps // 2
                init[:half, o + na:o + 2 * na] = 200.0
                init[:half, o + 3 * na] = 310.0
            lines = []
            jl = os.path.join(SCRATCH, "optim", f"{tag}_{regime}.jsonl")
            os.makedirs(os.path.dirname(jl), exist_ok=True)

            def log(row):
                lines.append(row)
                with open(jl, "a") as fh:
                    fh.write(json.dumps({k: (float(v) if k != "iter" else int(v))
                                         for k, v in row.items()}) + "\n")
                print(f"  {regime} it {row['iter']:3d} J {row['J']:+.6f} "
                      f"reward {row['reward']:+.6f} pen {row['penalty']:.3e} "
                      f"viol {row['viol']:.4f} |g| {row['gnorm']:.3e} "
                      f"{row['cpu']:.1f}s", flush=True)

            theta, hist, best = run_regime(
                plan, warm, regime, a.iters, a.lr, a.mu, log=log, init=init)
            np.savez_compressed(
                os.path.join(SCRATCH, "optim", f"{tag}_{regime}.npz"),
                acts=theta, runs=warm["runs"], hist=np.array(
                    [(h["iter"], h["J"], h["reward"], h["penalty"], h["viol"])
                     for h in hist]))
            exact = replay(sim, scenario, theta, warm["runs"],
                           offset_days=off, steps=a.steps)
            report["runs"][regime] = dict(
                history=hist, exact_total=float(exact.sum()),
                exact_mean=float(exact.mean()),
                exact_rewards=exact.tolist(),
                warm_total=float(warm["rewards"].sum()),
                wall_seconds=time.time() - t0)
            print(f"  {regime}: exact replay total {exact.sum():.6f} "
                  f"(warm {warm['rewards'].sum():.6f})", flush=True)
        with open(os.path.join(OUT, f"xopt_{tag}.json"), "w") as f:
            json.dump(report, f, indent=2)
        return 0

    print("compare: nothing to do")
    return 0


if __name__ == "__main__":
    # the C kernel's teardown corrupts the glibc heap on exit (observed as a
    # segfault after the results are written, WSL2/glibc only); skip it
    import sys as _sys
    _rc = main()
    _sys.stdout.flush()
    _sys.stderr.flush()
    os._exit(int(_rc))
