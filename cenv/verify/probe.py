"""Print the object graph a cenv component patches, for one scenario.

    starter-kit/.venv/bin/python -m cenv.verify.probe building_train_8/fairbanks_8 \
        --sim shipped [--steps 0]

Reports the zone order, the thermostat schedules the reward path queries, the
occupancy object, the reward function's constants, the cost tables and the
observation plumbing, plus the Python call counts per step of the functions the
bookkeeping component replaces. Everything printed is read from the live
environment, so it is also the reference for what a C port has to reproduce.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))


def scenario_config(sim: str, scenario: str) -> Path:
    building, weather = scenario.split("/")
    return (
        WORKSPACE / "sims" / sim / "smart_buildings" / "training_scenarios"
        / building / weather / "config.gin"
    )


def build_env(sim: str, scenario: str, start_offset_days: int = 0):
    """Load the environment exactly like sbx.evalrun does (no metrics writer)."""
    from sbx import simload

    simload.use_sim(sim)
    import smart_buildings  # noqa: F401  (root for the chdir)
    from smart_buildings import evaluate

    config = scenario_config(sim, scenario)
    if not config.exists():
        raise SystemExit(f"no config at {config}")
    env = evaluate.load_hybrid_environment(config)
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    return env, evaluate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--sim", default="shipped")
    ap.add_argument("--steps", type=int, default=0,
                    help="run N zero-action steps and report per-function call counts")
    args = ap.parse_args(argv)

    env, evaluate = build_env(args.sim, args.scenario)
    b = env.building
    sim = b.simulator
    hvac = sim.hvac
    print(f"env={type(env).__module__}.{type(env).__name__}")
    print(f"building={type(b).__module__}.{type(b).__name__}")
    print(f"simulator={type(sim).__module__}.{type(sim).__name__}")
    print(f"hvac={type(hvac).__module__}.{type(hvac).__name__}")

    rooms = b.building.room_dict if hasattr(b, "building") else sim.building.room_dict
    print(f"n_zones(room_dict)={len(rooms)} n_vavs={len(hvac.vavs)} "
          f"n_ahu=1 n_simulators=1")
    print(f"zone order[:6]={list(rooms)[:6]}")
    print(f"zone sizes[:6]={[len(rooms[z]) for z in list(rooms)[:6]]}")
    print(f"temp shape={sim.building.temp.shape} dtype={sim.building.temp.dtype} "
          f"contiguous={sim.building.temp.flags['C_CONTIGUOUS']}")

    scheds = {}
    for z, vavs in (hvac.zone_id_to_vav_ids.items() if hasattr(hvac, "zone_id_to_vav_ids") else []):
        pass
    ids = [id(v.thermostat.get_setpoint_schedule()) for v in hvac.vavs.values()]
    print(f"distinct schedules={len(set(ids))} over {len(ids)} vavs")
    s0 = next(iter(hvac.vavs.values())).thermostat.get_setpoint_schedule()
    print(f"schedule type={type(s0).__module__}.{type(s0).__name__}")
    for name in ("_heating_setpoint", "_cooling_setpoint", "_morning_start_hour",
                 "_evening_start_hour", "_time_zone", "_holidays"):
        if hasattr(s0, name):
            v = getattr(s0, name)
            print(f"  {name}={v!r}"[:160])

    occ = b.occupancy
    print(f"occupancy={type(occ).__module__}.{type(occ).__name__} "
          f"seed={getattr(occ, '_seed', None)}")

    rf = env.reward_function
    print(f"reward_function={type(rf).__module__}.{type(rf).__name__}")
    for name in ("_max_productivity_personhour_usd", "_min_productivity_personhour_usd",
                 "_productivity_deviation_ref", "_productivity_weight",
                 "_energy_cost_weight", "_carbon_emission_weight",
                 "_max_electricity_rate", "_max_natural_gas_rate"):
        print(f"  {name}={getattr(rf, name, '<missing>')}")
    ec = rf._electricity_energy_cost
    gc = rf._natural_gas_energy_cost
    print(f"elec cost={type(ec).__name__} weekday[0]={ec._weekday_energy_prices[0]!r} "
          f"weekend[0]={ec._weekend_energy_prices[0]!r} carbon[0]={ec._carbon_emission_rates[0]!r} "
          f"dtp={ec._weekday_energy_prices.dtype}")
    print(f"gas cost={type(gc).__name__} price[0]={gc._month_gas_price[0]!r} "
          f"carbon={gc._carbon_rate!r} dtp={gc._month_gas_price.dtype}")

    print(f"obs field_names={len(env.field_names)} "
          f"id_map={len(env._id_map)} normalizer={type(env._observation_normalizer).__name__} "
          f"hist_reducer={env._observation_histogram_reducer}")

    if args.steps:
        import numpy as np

        from smart_buildings.smart_control.simulator import building as building_mod

        counts: Counter = Counter()

        def wrap(obj, name, label=None):
            orig = getattr(obj, name)
            key = label or f"{type(obj).__name__}.{name}"

            def wrapper(*a, **k):
                counts[key] += 1
                return orig(*a, **k)

            setattr(obj, name, wrapper)

        wrap(sim.building, "get_zone_temp_stats")
        wrap(sim.building, "get_zone_average_temps")
        wrap(sim.building, "apply_thermal_power_zone")
        wrap(b.occupancy, "average_zone_occupancy", "occupancy.average_zone_occupancy")
        wrap(env, "_get_observation")
        wrap(env, "get_reward_info_and_response")
        wrap(env.reward_function, "compute_reward")
        wrap(env.reward_function, "_sum_zone_productivities")
        wrap(env.reward_function, "_sum_electricity_energy_rate")
        wrap(env.building.simulator, "reward_info")
        sched = next(iter(hvac.vavs.values())).thermostat.get_setpoint_schedule()
        for meth in ("get_temperature_window", "is_comfort_mode"):
            if hasattr(sched, meth):
                orig = getattr(sched, meth)

                def make(orig=orig, meth=meth):
                    def wrapper(*a, **k):
                        counts[f"schedule.{meth}"] += 1
                        return orig(*a, **k)
                    return wrapper

                setattr(sched, meth, make())
        obs, _ = env.reset()

        spec = env.action_spec()
        action = {
            "continuous_action": np.zeros(spec["continuous_action"].shape,
                                          dtype=np.float32),
            "discrete_action": np.zeros(spec["discrete_action"].shape,
                                        dtype=np.int32),
        }
        for _ in range(args.steps):
            env.step(action)
        print(f"calls over {args.steps} steps:")
        for key, v in sorted(counts.items()):
            print(f"  {key}: {v} ({v / args.steps:.1f}/step)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
