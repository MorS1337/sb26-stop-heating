"""Closed-loop check: the real environment with the thermal step replaced by the ROM.

Everything else is the shipped code -- CtrlW, the air handlers, the VAVs, the
hot water loop, the reward function, the occupancy model. Only
``FiniteDifferenceSolver.solve`` is replaced, so the reward, the observations and
the control decisions are computed exactly as the scored environment computes
them from whatever temperature field they are given.

This is the check that matters for using the surrogate in a training loop: the
controller closes the loop on the ROM, so its own actions respond to the ROM's
error, which damps it rather than letting it accumulate the way the open-loop
replay does. The ROM state is autonomous here (it never sees the field the
environment hands it), which is what a training environment would be.

Usage:
  starter-kit/.venv/bin/python -m fastsim.closedloop building_train_8/fairbanks_8 \
      --steps 2880 --m 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import scen
from .rom import Ops, ROM

DATA = Path(__file__).resolve().parent / "data"


def _zone_sum_matrix(o) -> "np.ndarray":
    """(n_solved x n_zones) 0/1 map, so a step's zone power is one sparse matvec.

    Doing this with a Python loop over zones re-derives each zone's coordinates
    from a list of tuples every step, which measured 19 ms/step — four times the
    reduced solve itself. The map is built once instead.
    """
    return o.Zind.tocsc()


def _zone_sums(o, scatter, solved, input_q):
    qflat = np.asarray(input_q, dtype=np.float64).reshape(-1)[solved]
    return scatter.T @ qflat


def run(
    scenario: str,
    m: int = 200,
    kind: str = "hybrid",
    steps: int = 2880,
    sim: str = "shipped",
    policy_spec: str = "policies.ctrl_w:CtrlW",
    snaps: np.ndarray | None = None,
    convection: bool = True,
    params: dict | None = None,
    rom_in=None,
    ops_in=None,
    use_rom: bool = True,
) -> dict:
    env, meta = scen.build_env(scenario, sim)
    simulator = env.building.simulator
    building = simulator.building
    if not convection:
        building._convection_simulator = None
    solver = simulator._solver
    building_name = scenario.split("/")[0]
    if rom_in is not None:
        rom, o = rom_in, ops_in
    else:
        o = Ops(building_name)
        rom = ROM(building_name, m=m, kind=kind, snaps=snaps, ops=o)
    zones = o.zones
    scatter = _zone_sum_matrix(o)
    solved = o.solved
    real_solve = solver.solve
    state = {"calls": 0}

    def solve_rom(temp, input_q, ambient_temperature, convection_coefficient,
                  air_side_conductance=None, supply_temperature=None):
        if state["calls"] == 0:
            rom.reset(np.asarray(temp, dtype=np.float64).reshape(-1), solver._mass_temp)
        state["calls"] += 1
        k = 0 if air_side_conductance is None else len(air_side_conductance)
        rom.set_ambient(ambient_temperature)
        rom.step(
            _zone_sums(o, scatter, solved, input_q),
            np.zeros(k) if k == 0 else air_side_conductance,
            np.zeros(k) if k == 0 else supply_temperature,
            ambient_temperature,
        )
        # The solver's own mass node is deliberately left alone: nothing in the
        # scored path reads it (only the solver's tests do), and rebuilding it
        # every step costs an n-sized np.where for no effect.
        return rom.field()

    if use_rom:
        solver.solve = solve_rom
    from smart_buildings import evaluate  # noqa: PLC0415

    policy_cls, policy = scen.make_policy(env, meta, policy_spec, params)
    obs, _ = env.reset()
    rewards, temps = [], []
    for _ in range(int(steps)):
        action = evaluate.validate_action(
            evaluate.get_policy_action(policy, obs), env.action_spec()
        )
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        temps.append(float(np.asarray(building.temp).mean()))
        if terminated or truncated:
            break
    solver.solve = real_solve
    return dict(
        mode="rom" if use_rom else "real",
        scenario=scenario,
        m=int(rom.W.shape[1]),
        kind=kind,
        steps=len(rewards),
        policy=policy_spec,
        params=params or {},
        convection=convection,
        total_reward=float(np.sum(rewards)),
        mean_per_step_reward=float(np.mean(rewards)),
        rewards=rewards,
        mean_temp=temps,
        rho_reduced=rom.rho,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scenarios", nargs="+")
    p.add_argument("--m", default="200")
    p.add_argument("--kinds", default="hybrid")
    p.add_argument("--steps", type=int, default=2880)
    p.add_argument("--sim", default="shipped")
    p.add_argument("--policy", default="policies.ctrl_w:CtrlW")
    p.add_argument("--snaps-from", default=None,
                   help="rollout dir whose snapshots seed the POD half of the basis")
    p.add_argument("--params", default=None, help="JSON dict of CtrlW PARAMS overrides")
    p.add_argument("--real-only", action="store_true",
                   help="leave the real solve in place (timing / reward reference)")
    p.add_argument("--no-convection", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    scen.pin_blas()
    scen.boot(args.sim)
    rows = []
    for scenario in args.scenarios:
        snaps = None
        src = args.snaps_from or str(Path(DATA) / scenario.replace("/", "__") / "convection")
        f = Path(src) / "snaps.npy"
        if f.exists():
            snaps = np.load(f)
        for kind in args.kinds.split(","):
            for m in [int(x) for x in args.m.split(",")]:
                r = run(scenario, m, kind, args.steps, args.sim, args.policy, snaps,
                        convection=not args.no_convection,
                        params=json.loads(args.params) if args.params else None,
                        use_rom=not args.real_only)
                rows.append(r)
                print(
                    f"{scenario} {kind} m={r['m']} steps={r['steps']} "
                    f"total_reward {r['total_reward']:.6f} "
                    f"mean {r['mean_per_step_reward']:.6f} "
                    f"| rho {r['rho_reduced']:.6f}"
                )
    if args.out:
        out = [dict(r, rewards=None, mean_temp=None) for r in rows]
        scen.resolve_out(args.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
