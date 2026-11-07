"""Cost of the reduced step, and what the real step costs, on one core.

Three numbers, all measured on this machine with one BLAS thread:

``real``  one SuperLU triangular solve (one RHS and a 30-wide block), the
          factorization, and the vector work around it. The real step does two
          such solves (``factorization.solve(forcing)`` and
          ``factorization.solve(P @ power)`` in _add_air_side), so the per-step
          solve cost is twice the single-RHS number. The end-to-end per-step
          CPU of a real CtrlW rollout is in ``fastsim/data/*/meta.json``.
``rom``   one ``ROM.step`` at a given m, plus the zone-mean readout.
``batch`` flops/bytes of the same step run for B environments of one building,
          which is how a training loop would actually use it.

Usage:
  starter-kit/.venv/bin/python -m fastsim.bench building_train_8 --m 100,400
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .rom import Ops, ROM


def bench_real(building: str, reps: int = 20) -> dict:
    o = Ops(building)
    rng = np.random.default_rng(0)
    one = rng.standard_normal(o.n)
    block = rng.standard_normal((o.n, o.P.shape[1]))
    t0 = time.perf_counter()
    for _ in range(reps):
        o.solve(one)
    t_one = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        o.solve(block)
    t_block = (time.perf_counter() - t0) / reps
    # vector work of one step: c*T + q + Tamb*(d+g) + mc*T_m, then the Woodbury
    # row (k^2 solve) and the field scatter for the exterior CVs.
    T = rng.standard_normal(o.n)
    q = rng.standard_normal(o.n)
    tm = rng.standard_normal(o.n)
    t0 = time.perf_counter()
    for _ in range(reps):
        forcing = o.c * T + q + 250.0 * (o.d + o.g_amb) + o.mc * tm
        out = np.full(o.nfull, 250.0)
        out[o.solved] = forcing
    t_vec = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    import scipy.sparse.linalg as spla

    spla.splu(o.A, permc_spec="MMD_AT_PLUS_A")
    t_factor = time.perf_counter() - t0
    return dict(
        building=building,
        n_solved=o.n,
        A_nnz=int(o.A.nnz),
        solve_1rhs_ms=1e3 * t_one,
        solve_block_ms=1e3 * t_block,
        block_width=int(o.P.shape[1]),
        forcing_vec_ms=1e3 * t_vec,
        factorize_ms=1e3 * t_factor,
        real_step_ms=1e3 * (2 * t_one + t_vec),
    )


def bench_rom(building: str, m: int, steps: int = 2000, kind: str = "hybrid",
              snaps: np.ndarray | None = None) -> dict:
    o = Ops(building)
    rom = ROM(building, m=m, kind=kind, snaps=snaps, ops=o)
    rng = np.random.default_rng(1)
    q = rng.uniform(0, 1e4, o.nz)
    G = np.full(o.P.shape[1], 300.0)
    Ts = np.full(o.P.shape[1], 298.0)
    rom.set_ambient(270.0)
    rom.step(q, G, Ts, 270.0)
    t0 = time.perf_counter()
    for _ in range(steps):
        rom.step(q, G, Ts, 270.0)
        rom.zone_means()
    t = (time.perf_counter() - t0) / steps
    mm = rom.W.shape[1]
    flops = 2 * mm * mm + 2 * mm * o.nz + 2 * mm * o.ng + 2 * mm * o.P.shape[1] \
        + 2 * o.P.shape[1] * mm + o.P.shape[1] ** 3 / 3 + 2 * mm * o.nz
    return dict(
        building=building,
        m_requested=m,
        m=mm,
        kind=kind,
        step_ms=1e3 * t,
        env_steps_per_single_core_s=1.0 / t,
        rom_flops_per_step=int(flops),
        state_bytes=int(mm * 8 + o.ng * 8),
    )


def batch_projection(building: str, m: int, envs: int) -> dict:
    """Flops and bytes for a batched step of ``envs`` copies of one building."""
    o = Ops(building)
    k = o.P.shape[1]
    mm = m
    flops = envs * (
        2 * mm * mm + 2 * mm * o.nz + 2 * mm * o.ng + 2 * mm * k + 2 * k * mm + k ** 3
    )
    bytes_ = envs * 8 * (mm + o.ng) + 8 * (mm * mm + mm * o.nz + mm * o.ng
                                          + mm * k + k * mm + k * k)
    return dict(building=building, envs=envs, m=m, flops_per_step=flops,
                bytes_per_step=int(bytes_), arithmetic_intensity=flops / bytes_)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("buildings", nargs="+")
    p.add_argument("--m", default="100,400")
    p.add_argument("--envs", default="1,64,1024")
    p.add_argument("--steps", type=int, default=2000)
    args = p.parse_args(argv)
    rows = {"real": [], "rom": [], "batch": []}
    for b in args.buildings:
        rows["real"].append(bench_real(b))
        for m in [int(x) for x in args.m.split(",")]:
            rows["rom"].append(bench_rom(b, m, args.steps))
        for e in [int(x) for x in args.envs.split(",")]:
            rows["batch"].append(batch_projection(b, max(int(x) for x in args.m.split(",")), e))
    for r in rows["real"]:
        print(
            f"real {r['building']}: solve1 {r['solve_1rhs_ms']:.3f} ms, "
            f"block{r['block_width']} {r['solve_block_ms']:.3f} ms, "
            f"forcing {r['forcing_vec_ms']:.3f} ms, factorize {r['factorize_ms']:.0f} ms "
            f"=> step {r['real_step_ms']:.2f} ms"
        )
    for r in rows["rom"]:
        print(
            f"rom  {r['building']} m={r['m']:3d} {r['kind']}: {r['step_ms']:.3f} ms/step "
            f"({r['env_steps_per_single_core_s']:.0f} env-steps/s per core), "
            f"{r['rom_flops_per_step']/1e3:.0f} kflop/step"
        )
    for r in rows["batch"]:
        print(
            f"batch {r['building']} B={r['envs']:4d} m={r['m']}: "
            f"{r['flops_per_step']/1e9:.3f} GFLOP/step, {r['bytes_per_step']/1e6:.2f} MB, "
            f"AI {r['arithmetic_intensity']:.0f} flop/byte"
        )
    print(json.dumps(rows["real"] + rows["rom"] + rows["batch"], indent=2)[:0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
