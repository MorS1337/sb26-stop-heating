"""Fuzz gate for the HVAC device ports.

The episode-shadow gate (``python -m cenv.shadow --component hvac``) only
exercises the methods the flexible-floor-plan path actually calls; the coil
discharge path (``compute_zone_supply_temp``, ``compute_energy_applied_to_zone``)
belongs to the older ``Simulator`` and never runs under ``--sim shipped`` there.
Those are pure functions of device state, so they are gated here instead: every
ported method is called on the live device objects with a few thousand draws
from its real argument range, beside the untouched Python original.

    starter-kit/.venv/bin/python -m cenv.verify.hvac_gate \
        --sim shipped --scenario building_train_8/fairbanks_8 --draws 4000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]

REL_TOL = 1e-12


def _instances(env):
    hvac = env.building.simulator._hvac
    ahu_system = hvac.air_handler
    ahus = getattr(ahu_system, "ahus", [ahu_system])
    vav = next(iter(hvac.vavs.values()))
    hws = hvac.hot_water_system
    return {
        "ahu": ahus[0],
        "vav": vav,
        "boiler": hws._heat_source,
        "hws": hws,
        "pump": hws._pump,
    }


def _samplers(rng, inst, n):
    """Per method: a list of (setup(obj), args) draws from its real range.

    ``setup`` is what the method reads off the device (recirculation fraction,
    tank state) rather than off its arguments; it is applied before each of the
    two calls so both sides see the same state.
    """
    ahu = inst["ahu"]
    nop = lambda obj: None

    def temps(n):
        return rng.uniform(250.0, 320.0, n)

    def rep(count, fancy):
        out = []
        for _ in range(count):
            out.append(fancy())
        return out

    def ahu_draw():
        r = float(rng.uniform(0.0, float(getattr(ahu, "_max_recirculation", 0.7))))
        t = float(rng.uniform(250.0, 320.0))
        a = float(rng.uniform(250.0, 320.0))
        return ((lambda obj, r=r: setattr(obj, "_recirculation", r)), (t, a))

    out = {}
    for name in (
        "AirHandler.get_mixed_air_temp",
        "AirHandler.get_supply_air_temp",
        "AirHandler.compute_thermal_energy_rate",
        "AirHandler.compute_cooling_electrical_energy_rate",
    ):
        out[name] = rep(n, ahu_draw)

    out["AirHandler.compute_fan_power"] = rep(n, lambda: (
        nop,
        (float(rng.uniform(0, 10)), float(rng.uniform(0, 2500)),
         float(rng.uniform(0.5, 1.0))),
    ))
    out["AirHandler.compute_intake_fan_energy_rate"] = rep(n, lambda: (nop, ()))
    out["AirHandler.compute_exhaust_fan_energy_rate"] = rep(n, lambda: (nop, ()))
    out["Vav._compute_flow_rate_demand"] = rep(n, lambda: (nop, ()))
    out["Vav._reheat_effectiveness"] = rep(n, lambda: (
        nop, () if rng.random() < 0.5 else
        (float(rng.uniform(250, 320)), float(rng.uniform(250, 350))),
    ))
    out["Vav.compute_reheat_energy_rate"] = rep(n, lambda: (
        nop, (float(rng.uniform(250, 320)), float(rng.uniform(250, 350))),
    ))
    out["Vav.compute_zone_supply_temp"] = rep(n, lambda: (
        nop, (float(rng.uniform(250, 320)), float(rng.uniform(250, 350))),
    ))
    out["Vav.compute_energy_applied_to_zone"] = rep(n, lambda: (
        nop,
        (float(rng.uniform(250, 320)), float(rng.uniform(250, 320)),
         float(rng.uniform(250, 350))),
    ))
    out["Boiler.compute_thermal_dissipation_rate"] = rep(n, lambda: (
        nop, (float(rng.uniform(250, 350)), float(rng.uniform(230, 320))),
    ))

    def boiler_state_draw():
        # The state has to be drawn once per draw, not once per setup call:
        # both sides must be handed the same tank.
        temp = float(rng.uniform(300.0, 350.0))
        change = float(rng.uniform(-2.0, 2.0))
        duration = _td(float(rng.uniform(0.0, 900.0)))

        def setup(obj):
            obj._current_temperature = temp
            obj._step_tank_temperature_change = change
            obj._last_step_duration = duration

        return (setup, (float(rng.uniform(280, 350)),
                        float(rng.uniform(230, 320)),
                        float(rng.uniform(0, 0.01))))

    out["Boiler.compute_thermal_energy_rate"] = rep(n, boiler_state_draw)
    out["Boiler._adjust_temperature"] = rep(n, lambda: (
        nop,
        (float(rng.uniform(280, 350)), float(rng.uniform(280, 350)),
         _td(float(rng.uniform(0.0, 900.0)))),
    ))
    out["HotWaterSystem._calculate_flow_rate"] = rep(n, lambda: (
        nop, (float(rng.uniform(0, 2)), float(rng.uniform(0, 50))),
    ))
    out["HotWaterSystem.compute_pump_power"] = rep(n, lambda: (nop, ()))
    out["WaterPump.compute_pump_power"] = rep(n, lambda: (
        nop, (float(rng.uniform(0, 0.01)),),
    ))
    out["WaterPump._convert_differential_head_to_pressure"] = rep(n, lambda: (
        nop, (float(rng.uniform(0, 40)),),
    ))
    out["WaterPump._convert_pressure_to_differential_head"] = rep(n, lambda: (
        nop, (float(rng.uniform(0, 400000)),),
    ))
    def heat_draw_draw():
        start = float(rng.uniform(300.0, 350.0))

        def setup(obj):
            obj._current_temperature = start
            obj._burner_heat_rate = 0.0

        return (setup, (float(rng.uniform(0, 5e5)),
                        float(rng.uniform(230, 320)),
                        float(rng.uniform(1.0, 900.0))))

    out["Boiler.apply_heat_draw"] = rep(n, heat_draw_draw)
    return out


def _td(seconds: float):
    import pandas as pd

    return pd.Timedelta(seconds, unit="s")


def compare(env, draws: int, seed: int = 0) -> list[dict]:
    from cenv import hvac as hv  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    inst = _instances(env)
    samplers = _samplers(rng, inst, draws)
    rows = []
    for cls, name, c_fn in hv._patches():
        original = hv._ORIGINALS.get((cls, name), cls.__dict__.get(name))
        if original is None:
            original = getattr(cls, name)
        label = f"{cls.__name__}.{name}"
        sample = samplers.get(label)
        if sample is None:
            continue
        try:
            obj = {
                "AirHandler": inst["ahu"], "Vav": inst["vav"],
                "Boiler": inst["boiler"], "HotWaterSystem": inst["hws"],
                "WaterPump": inst["pump"],
            }[cls.__name__]
        except KeyError:
            continue
        pairs = sample
        max_abs = 0.0
        max_rel = 0.0
        bad = 0
        exact = 0
        returns_state = label.endswith("apply_heat_draw")

        def outputs(value):
            """The compared quantities: the return value, or the mutated state."""
            if not returns_state:
                return [float(value)]
            return [float(obj._current_temperature),
                    float(getattr(obj, "_burner_heat_rate", 0.0))]

        for setup, args in pairs:
            setup(obj)
            py = outputs(original(obj, *args))
            tank = None
            if hasattr(obj, "_current_temperature"):
                tank = (obj._current_temperature,
                        getattr(obj, "_burner_heat_rate", 0.0))
            setup(obj)
            c = outputs(c_fn(obj, *args))
            if tank is not None:
                obj._current_temperature = tank[0]
                if hasattr(obj, "_burner_heat_rate"):
                    obj._burner_heat_rate = tank[1]
            for a, b in zip(py, c):
                diff = abs(b - a)
                if diff == 0.0:
                    exact += 1
                    continue
                scale = max(abs(a), abs(b))
                rel = diff / scale if scale > 0.0 else float("inf")
                max_abs = max(max_abs, diff)
                max_rel = max(max_rel, rel)
                if rel > REL_TOL:
                    bad += 1
        rows.append({
            "label": label, "calls": len(pairs),
            "exact_fraction": exact / (len(pairs) * (2 if returns_state else 1)),
            "max_abs": max_abs, "max_rel": max_rel, "n_bad": bad,
            "passed": bad == 0 and bool(pairs),
        })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim", default="shipped")
    ap.add_argument("--scenario", default="building_train_8/fairbanks_8")
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if str(WORKSPACE) not in sys.path:
        sys.path.insert(0, str(WORKSPACE))
    from cenv.verify.probe import build_env  # noqa: PLC0415
    from cenv import hvac as hv  # noqa: PLC0415

    env, _evaluate = build_env(args.sim, args.scenario)
    # Patch first so _ORIGINALS holds the untouched Python methods.
    report = hv.install()
    rows = compare(env, args.draws, args.seed)
    width = max(len(r["label"]) for r in rows)
    print(f"[hvac fuzz {args.sim}] {report['patched'].__len__()} patched methods, "
          f"{args.draws} draws each, rel_tol={REL_TOL:g}")
    print(f"{'quantity'.ljust(width)}  calls  exact%    max_abs    max_rel  status")
    for r in rows:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{r['label'].ljust(width)}  {r['calls']:5d}  "
              f"{100 * r['exact_fraction']:6.1f}  {r['max_abs']:9.2e}  "
              f"{r['max_rel']:9.2e}  {status}")
    ok = all(r["passed"] for r in rows)
    print(f"[hvac fuzz {args.sim}] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
