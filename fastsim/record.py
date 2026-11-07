"""Instrumented CtrlW rollout: logs everything the thermal solve consumes.

The wrapper sits on the solver instance, so it sees the exact arrays the real
``solve`` was called with -- after the air handler has capped flows, after the
reheat was distributed, after internal gains and occupants were added -- and the
exact field it returned. Nothing is recomputed from policy state.

Logged per step
---------------
``q_zone``   (S, n_zones) W, the total power the step applied to each zone.
             ``apply_thermal_power_zone`` (building.py:1097-1123) writes
             ``power / n_cv`` to every CV of the zone, so this column plus the
             zone geometry reproduces ``input_q`` exactly; the recorder asserts
             that every step (``q_uniform_err``).
``G``        (S, k) W/K, the air-side conductance of each VAV
             = flow_rate_demand * rho * cp (simulator_flexible_floor_plan.py:418)
``Ts``       (S, k) K, supply air temperature seen by each VAV
``Tamb``     (S,) K, ``h`` (S,) W/m2K
``ZT_in``    (S, n_zones) zone means entering the step
``ZT_out``   (S, n_zones) zone means leaving it (post-convection would be later;
             this is what the solve produced)
``snaps``    full fields every ``snap_every`` steps, float32, for the POD basis
``reheat``   (S,) W, total reheat coil duty the step charged (temperature
             independent; kept because energy is computed from it)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import scen

DATA = Path(__file__).resolve().parent / "data"


def record(
    scenario: str,
    steps: int = 2880,
    sim: str = "shipped",
    convection: bool = True,
    snap_every: int = 4,
    policy_spec: str = "policies.ctrl_w:CtrlW",
    out_root: Path | None = None,
) -> Path:
    env, meta = scen.build_env(scenario, sim)
    simulator = env.building.simulator
    building = simulator.building
    if not convection:
        building._convection_simulator = None
    solver = simulator._solver

    from .extract import physical_zones  # noqa: PLC0415

    zones = physical_zones(building)
    shape = building.temp.shape
    ncols = shape[1]
    zcoords = {
        z: np.asarray(building.room_dict[z], dtype=np.intp).reshape(-1, 2)
        for z in zones
    }
    zflat = [zcoords[z][:, 0] * ncols + zcoords[z][:, 1] for z in zones]
    zcount = np.array([f.size for f in zflat], dtype=np.float64)

    policy_cls, policy = scen.make_policy(env, meta, policy_spec)

    log = {k: [] for k in ("q_zone", "G", "Ts", "Tamb", "h", "ZT_in", "ZT_out", "reheat")}
    rewards: list[float] = []
    snaps: list[np.ndarray] = []
    uniform = [0.0]

    original = solver.solve

    def solve_logged(
        temp,
        input_q,
        ambient_temperature,
        convection_coefficient,
        air_side_conductance=None,
        supply_temperature=None,
    ):
        out = original(
            temp,
            input_q,
            ambient_temperature,
            convection_coefficient,
            air_side_conductance,
            supply_temperature,
        )
        qflat = np.asarray(input_q, dtype=np.float64).reshape(-1)
        qz = np.array([qflat[f].sum() for f in zflat])
        worst = max(
            float(np.max(np.abs(qflat[f] - q / f.size))) if f.size else 0.0
            for f, q in zip(zflat, qz)
        )
        uniform[0] = max(uniform[0], worst)
        log["q_zone"].append(qz)
        k = 0 if air_side_conductance is None else len(air_side_conductance)
        log["G"].append(
            np.zeros(k) if k == 0 else np.asarray(air_side_conductance, float).copy()
        )
        log["Ts"].append(
            np.zeros(k) if k == 0 else np.asarray(supply_temperature, float).copy()
        )
        log["Tamb"].append(float(ambient_temperature))
        log["h"].append(float(convection_coefficient))
        tflat = np.asarray(temp, dtype=np.float64).reshape(-1)
        oflat = np.asarray(out, dtype=np.float64).reshape(-1)
        log["ZT_in"].append(np.array([tflat[f].mean() for f in zflat]))
        log["ZT_out"].append(np.array([oflat[f].mean() for f in zflat]))
        if len(snaps) * snap_every <= len(log["Tamb"]) - 1:
            if (len(log["Tamb"]) - 1) % snap_every == 0:
                snaps.append(oflat.astype(np.float32))
        return out

    solver.solve = solve_logged
    obs, _ = env.reset()
    started = time.time()
    cpu0 = time.process_time()
    reheat = []
    from smart_buildings import evaluate  # noqa: PLC0415

    for _ in range(int(steps)):
        action = evaluate.validate_action(
            evaluate.get_policy_action(policy, obs), env.action_spec()
        )
        try:
            reheat.append(float(env.building.simulator._hvac.hot_water_system.total_heating_rate))
        except Exception:  # noqa: BLE001 - diagnostic only
            reheat.append(0.0)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        if terminated or truncated:
            break
    wall = time.time() - started
    cpu = time.process_time() - cpu0
    solver.solve = original

    out = (out_root or DATA) / scenario.replace("/", "__") / (
        "convection" if convection else "noconv"
    )
    out.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in log.items() if v}
    arrays["zone_count"] = zcount
    arrays["rewards"] = np.asarray(rewards, dtype=np.float64)
    arrays["snap_every"] = np.array(snap_every)
    np.savez_compressed(out / "rollout.npz", **arrays)
    np.save(out / "snaps.npy", np.stack(snaps) if snaps else np.zeros((0, 1), np.float32))
    (out / "meta.json").write_text(
        json.dumps(
            dict(
                scenario=scenario,
                sim=sim,
                convection=convection,
                steps=int(len(log["Tamb"])),
                grid=list(shape),
                zones=zones,
                snap_every=snap_every,
                n_snaps=len(snaps),
                q_uniform_err=uniform[0],
                total_reward=float(np.sum(rewards)),
                policy=policy_spec,
                wall_seconds=wall,
                cpu_seconds=cpu,
                step_cpu_seconds=cpu / max(1, len(log["Tamb"])),
            ),
            indent=2,
        )
    )
    print(
        f"{scenario} convection={convection}: {len(log['Tamb'])} steps in {cpu:.1f} CPU-s "
        f"({cpu/max(1,len(log['Tamb']))*1000:.1f} ms/step), snaps {len(snaps)}, "
        f"q_uniform_err {uniform[0]:.3g} -> {out}"
    )
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scenarios", nargs="+")
    p.add_argument("--steps", type=int, default=2880)
    p.add_argument("--sim", default="shipped")
    p.add_argument("--no-convection", action="store_true")
    p.add_argument("--snap-every", type=int, default=4)
    p.add_argument("--policy", default="policies.ctrl_w:CtrlW")
    args = p.parse_args(argv)
    scen.pin_blas()
    scen.boot(args.sim)
    for s in args.scenarios:
        record(
            s,
            args.steps,
            args.sim,
            convection=not args.no_convection,
            snap_every=args.snap_every,
            policy_spec=args.policy,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
